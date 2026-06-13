"""
train.py for Go2 quadruped locomotion (Stable-Baselines3 PPO + MuJoCo).

학습 전용 스크립트. 추론/롤아웃/영상 녹화는 test.py 가 담당한다.
Go2MujocoEnv 는 position 제어 전용(unitree_go2/scene_position.xml)이므로
torque/ctrl_type 관련 옵션은 다루지 않는다. CLI 인자는 모두 선택값이며,
미지정 시 params.yaml 값을 사용한다.
"""

import os
# 재현(determinism)용: deterministic cuBLAS workspace. 반드시 torch import 전에 설정해야 함.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from go2_mujoco_env import Go2MujocoEnv
from utils.reward_logging_callback import RewardLoggingCallback


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def enable_determinism(seed: int) -> None:
    """가능한 한 재현 가능하게 만드는 설정.

    주의: GPU + SubprocVecEnv 환경에서는 완벽한 비트 단위 재현이 보장되지 않으며,
    deterministic 알고리즘은 학습을 다소 느리게 만든다. 일부 연산은 deterministic
    구현이 없어 warn_only=True 로 경고만 내고 진행한다.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def load_config(base_dir: Path) -> dict:
    cfg_path = base_dir / "src" / "params.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"params.yaml not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 필수 키 검증 (빠진 키를 조기에 알려줌)
    required_top = ["n_envs", "seed", "eval_freq", "total_timestep"]
    required_policy = [
        "use_pretrained", "learning_rate", "n_steps", "batch_size", "n_epochs",
        "gamma", "gae_lambda", "clip_range", "normalize_advantage",
        "ent_coef", "vf_coef", "max_grad_norm",
    ]
    for k in required_top:
        if k not in cfg:
            raise KeyError(f"params.yaml 최상위 키 누락: '{k}'")
    for k in required_policy:
        if k not in cfg.get("policy", {}):
            raise KeyError(f"params.yaml policy 키 누락: 'policy.{k}'")
    if "interval" not in cfg.get("log", {}):
        raise KeyError("params.yaml 키 누락: 'log.interval'")
    return cfg


def build_ppo_kwargs(cfg: dict) -> dict:
    p = cfg["policy"]
    return dict(
        learning_rate=p["learning_rate"],
        n_steps=p["n_steps"],
        batch_size=p["batch_size"],
        n_epochs=p["n_epochs"],
        gamma=p["gamma"],
        gae_lambda=p["gae_lambda"],
        clip_range=p["clip_range"],
        normalize_advantage=p["normalize_advantage"],
        ent_coef=p["ent_coef"],
        vf_coef=p["vf_coef"],
        max_grad_norm=p["max_grad_norm"],
    )


def resolve_pretrained_path(cfg: dict, base_dir: Path, cli_path: str | None) -> Path | None:
    """우선순위: CLI --model_path > yaml policy.pretrained_path (use_pretrained=True일 때)."""
    if cli_path:
        return Path(cli_path)
    if cfg["policy"].get("use_pretrained"):
        yaml_path = cfg["policy"].get("pretrained_path")
        if not yaml_path:
            raise ValueError(
                "use_pretrained=True 이지만 'policy.pretrained_path'가 yaml에 없습니다."
            )
        return (base_dir / yaml_path) if not Path(yaml_path).is_absolute() else Path(yaml_path)
    return None


def make_env_kwargs(base_dir: Path, render_mode=None) -> dict:
    """Go2MujocoEnv 생성 인자 (position 제어 전용)."""
    return {"prj_path": base_dir.as_posix(), "render_mode": render_mode}


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def train(args):
    base_dir = Path(__file__).resolve().parents[1]
    cfg = load_config(base_dir)

    model_dir = base_dir / "models"
    log_dir = base_dir / "logs"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    n_envs = args.num_parallel_envs or cfg["n_envs"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    total_timesteps = args.total_timesteps or cfg["total_timestep"]

    if args.deterministic:
        enable_determinism(seed)
        print(f"[INFO] deterministic 모드 ON (seed={seed}). "
              f"GPU/SubprocVecEnv 에서는 완전 재현은 보장되지 않으며 다소 느려질 수 있습니다.")

    # n_steps * n_envs 가 batch_size 로 나누어떨어지는지 확인 (미니배치 truncation 방지)
    rollout = cfg["policy"]["n_steps"] * n_envs
    if rollout % cfg["policy"]["batch_size"] != 0:
        print(
            f"[WARN] (n_steps*n_envs)={rollout} 가 batch_size="
            f"{cfg['policy']['batch_size']} 로 나누어떨어지지 않습니다. 미니배치 일부가 버려집니다."
        )

    train_env_kwargs = make_env_kwargs(base_dir, render_mode=None)
    vec_env = make_vec_env(
        Go2MujocoEnv,
        env_kwargs=train_env_kwargs,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
    )

    # 평가 전용 env (학습 env와 분리, 단일 환경, seed 다르게)
    eval_env = make_vec_env(
        Go2MujocoEnv,
        env_kwargs=train_env_kwargs,
        n_envs=1,
        seed=seed + 10_000,
        vec_env_cls=DummyVecEnv,
    )

    try:
        print(f"[INFO] base_dir = {base_dir}")
        print(f"[INFO] n_envs = {n_envs}, seed = {seed}, total_timesteps = {total_timesteps}")
        print(f"[INFO] Action space: {vec_env.action_space}")
        print(f"[INFO] Observation space: {vec_env.observation_space}")

        train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        run_name = train_time if args.run_name is None else f"{train_time}-{args.run_name}"
        model_path = model_dir / run_name
        model_path.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving models to '{model_path}'")

        save_freq_steps = cfg["policy"]["n_steps"] * cfg["log"]["interval"]
        checkpoint_callback = CheckpointCallback(
            save_freq=max(save_freq_steps // n_envs, 1),
            save_path=str(model_path),
            name_prefix="model",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_path),
            log_path=str(log_dir),
            eval_freq=max(cfg["eval_freq"] // n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        )
        reward_logging_callback = RewardLoggingCallback()
        callbacks = CallbackList(
            [eval_callback, checkpoint_callback, reward_logging_callback]
        )

        ppo_kwargs = build_ppo_kwargs(cfg)
        pretrained_path = resolve_pretrained_path(cfg, base_dir, args.model_path)

        if pretrained_path is not None:
            if not Path(pretrained_path).exists():
                raise FileNotFoundError(f"pretrained model not found: {pretrained_path}")
            print(f"[INFO] Loading pretrained model from {pretrained_path}")
            model = PPO.load(
                str(pretrained_path),
                env=vec_env,
                verbose=1,
                tensorboard_log=str(log_dir),
            )
            model.learning_rate = cfg["policy"]["learning_rate"]
            model._setup_lr_schedule()
        else:
            model = PPO(
                "MlpPolicy",
                env=vec_env,
                verbose=1,
                tensorboard_log=str(log_dir),
                seed=seed,
                **ppo_kwargs,
            )

        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=(pretrained_path is None),
            progress_bar=True,
            tb_log_name=run_name,
            callback=callbacks,
        )
        model.save(model_path / "final_model")
        print(f"[INFO] Final model saved to {model_path / 'final_model'}")
    finally:
        vec_env.close()
        eval_env.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Go2 PPO locomotion policy.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="실행 이름. models/ 아래에 학습 시각이 접두로 붙어 저장됨.")
    parser.add_argument("--num_parallel_envs", type=int, default=None,
                        help="병렬 환경 수 (미지정 시 yaml n_envs 사용)")
    parser.add_argument("--total_timesteps", type=int, default=None,
                        help="학습 총 timestep (미지정 시 yaml total_timestep 사용)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="재개용 시작 모델 (.zip). 미지정 시 처음부터 학습")
    parser.add_argument("--seed", type=int, default=None,
                        help="미지정 시 yaml seed 사용")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="재현 가능 모드. --no-deterministic 로 끄면 더 빠르지만 런마다 결과가 달라짐")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
