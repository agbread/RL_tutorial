import argparse
import time
from pathlib import Path

import yaml
import mujoco
import imageio
from tqdm.auto import tqdm

if not hasattr(mujoco.MjData, "solver_iter"):
    setattr(mujoco.MjData, "solver_iter", property(lambda self: self.solver_niter))


import sys
import numpy
import numpy.core, numpy.core.numeric, numpy.core.multiarray
sys.modules.setdefault("numpy.core_", numpy.core)
sys.modules.setdefault("numpy.core_.numeric", numpy.core.numeric)
sys.modules.setdefault("numpy.core_.multiarray", numpy.core.multiarray)

from stable_baselines3 import PPO
from go2_mujoco_env import Go2MujocoEnv


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_config(base_dir: Path) -> dict:
    cfg_path = base_dir / "src" / "params.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"params.yaml not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_model_path(base_dir: Path, args) -> Path:
    if args.model_path:
        p = Path(args.model_path)
        return p if p.is_absolute() else base_dir / p
    return base_dir / "models" / args.model_name / args.model_file


def resolve_command(args, test_cfg: dict) -> list:
    """우선순위: --command > yaml test.given_command > 기본값."""
    if args.command is not None:
        return list(args.command)
    return list(test_cfg.get("given_command", [0.9, 0.0, 0.0]))


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #
def test(args):
    base_dir = Path(__file__).resolve().parents[1]
    cfg = load_config(base_dir)
    test_cfg = cfg.get("test", {})

    model_path = resolve_model_path(base_dir, args)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    # max_time_step_s: CLI > yaml (없으면 에러)
    max_time_step_s = args.max_time_step_s
    if max_time_step_s is None:
        if "max_time_step_s" not in test_cfg:
            raise KeyError(
                "max_time_step_s 가 --max_time_step_s 로도, params.yaml 'test.max_time_step_s' 로도 "
                "지정되지 않았습니다."
            )
        max_time_step_s = test_cfg["max_time_step_s"]

    given_command = resolve_command(args, test_cfg)
    control_hz = args.control_hz
    video_fps = args.video_fps
    deterministic = not args.stochastic

    env = Go2MujocoEnv(
        prj_path=base_dir.as_posix(),
        given_command=given_command,
        render_mode="rgb_array",
        camera_name=args.camera_name,
        width=args.width,
        height=args.height,
    )

    run_name = model_path.parent.name
    video_path = base_dir / "models" / run_name / f"rollout_{run_name}.mp4"

    render_interval = max(control_hz // video_fps, 1)
    max_steps = int(max_time_step_s * control_hz)

    print(f"[INFO] model        : {model_path}")
    print(f"[INFO] command      : {given_command}")
    print(f"[INFO] control_hz   : {control_hz}, video_fps: {video_fps}")
    print(f"[INFO] max_time_s   : {max_time_step_s}  -> max_steps: {max_steps}")
    print(f"[INFO] deterministic: {deterministic}")

    t_render = 0.0
    n_render = 0
    last_render = 0.0

    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
    }

    try:
        model = PPO.load(path=str(model_path), env=env, verbose=1,
                         custom_objects=custom_objects)

        frames = []
        pbar = tqdm(total=max_steps, desc="rollout", unit="step", dynamic_ncols=True)

        reset_kwargs = {"seed": args.seed} if args.seed is not None else {}
        obs, _ = env.reset(**reset_kwargs)
        start = time.perf_counter()

        ep_len = 0
        ep_reward = 0.0
        total_reward = 0.0
        n_episodes = 0

        for global_step in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_len += 1
            total_reward += reward

            if args.sleep > 0:
                time.sleep(args.sleep)

            if not args.no_video and global_step % render_interval == 0:
                t0 = time.perf_counter()
                frame = env.render()
                frames.append(frame)
                last_render = time.perf_counter() - t0
                t_render += last_render
                n_render += 1

            elapsed = time.perf_counter() - start
            steps_per_sec = (global_step + 1) / max(elapsed, 1e-9)
            avg_render = (t_render / n_render) if n_render else 0.0
            pbar.set_postfix({
                "steps/s": f"{steps_per_sec:6.1f}",
                "renders": n_render,
                "r_last(s)": f"{last_render:5.3f}",
                "r_avg(s)": f"{avg_render:5.3f}",
            })
            pbar.update(1)

            if terminated or truncated:
                n_episodes += 1
                print(f"episode finished: ep_len={ep_len}, ep_reward={ep_reward:.3f}")
                obs, _ = env.reset()
                ep_len = 0
                ep_reward = 0.0

        pbar.close()
        print(f"[INFO] total_reward over rollout: {total_reward:.3f} "
              f"(completed episodes: {n_episodes})")

        if not args.no_video:
            if frames:
                imageio.mimwrite(
                    video_path,
                    frames,
                    fps=video_fps,
                    codec="libx264",
                    quality=8,
                    pixelformat="yuv420p",
                )
            else:
                print("[WARN] 렌더된 프레임이 없어 비디오를 저장하지 않습니다.")

    finally:
        env.close()
        print("avg render sec:", t_render / max(n_render, 1))
        if not args.no_video:
            print("Saved video to:", video_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Rollout a trained Go2 PPO model and record video.")
    # 모델 지정
    parser.add_argument("--model_path", type=str, default=None,
                        help="모델 zip 전체 경로 (지정 시 --model_name/--model_file 무시)")
    parser.add_argument("--model_name", type=str, default="Go2_forth_test",
                        help="models/<model_name>/ 아래에서 모델을 찾음")
    parser.add_argument("--model_file", type=str, default="best_model.zip",
                        help="모델 파일 이름 (예: best_model.zip, final_model.zip)")
    # 명령 / 환경
    parser.add_argument("--command", type=float, nargs=3, default=None,
                        metavar=("VX", "VY", "WZ"),
                        help="desired velocity command [vx vy wz] (미지정 시 yaml/기본값)")
    parser.add_argument("--camera_name", type=str, default="tracking")
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=1440)
    # 시간 / 속도
    parser.add_argument("--max_time_step_s", type=float, default=None,
                        help="롤아웃 시간(초). 미지정 시 yaml test.max_time_step_s 사용")
    parser.add_argument("--control_hz", type=int, default=50,
                        help="제어 주파수(Hz). max_steps = max_time_step_s * control_hz")
    parser.add_argument("--video_fps", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.02,
                        help="스텝 간 sleep(초). 0이면 최대 속도")
    # 동작 옵션
    parser.add_argument("--stochastic", action="store_true",
                        help="설정 시 deterministic=False 로 추론")
    parser.add_argument("--no_video", action="store_true",
                        help="비디오 저장/렌더 생략")
    parser.add_argument("--seed", type=int, default=None,
                        help="첫 reset 시드")
    return parser.parse_args()


if __name__ == "__main__":
    test(parse_args())
