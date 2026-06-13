"""격리 렌더링 스크립트.

Colab에서 학습 직후 같은 커널에서 env.render() 를 호출하면 (CUDA/EGL/잔여
서브프로세스 상태 충돌로) 세션이 통째로 죽는 경우가 있다. 이를 피하기 위해
이 스크립트를 별도 프로세스로 실행한다:
  - 깨끗한 새 프로세스 (학습 커널의 누적 상태 없음)
  - 모델을 CPU 로 로드 (CUDA ↔ EGL 충돌 회피, MLP 추론은 CPU 로 충분히 빠름)
  - torch 를 mujoco 보다 먼저 import (네이티브 로드 순서 충돌 회피)
서브프로세스가 죽어도 노트북 커널은 살아있다.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prj", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--command", nargs=3, type=float, default=[0.9, 0.0, 0.0])
    ap.add_argument("--max_time_s", type=float, default=4.0)
    ap.add_argument("--control_hz", type=int, default=50)
    ap.add_argument("--video_fps", type=int, default=10)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--camera", default="tracking")
    ap.add_argument("--gl", default="egl")
    a = ap.parse_args()

    os.environ["MUJOCO_GL"] = a.gl
    sys.path.insert(0, a.prj)
    sys.path.insert(0, os.path.join(a.prj, "src"))

    import torch  # noqa: F401  (mujoco 보다 먼저)
    import imageio
    from stable_baselines3 import PPO
    import src.go2_mujoco_env as go2_env

    env = go2_env.Go2MujocoEnv(
        prj_path=a.prj, cfg_path=a.cfg, given_command=a.command,
        render_mode="rgb_array", camera_name=a.camera,
        width=a.width, height=a.height,
    )
    env._reset_noise_scale = 0.05
    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "lr_schedule": lambda _: 1e-4,   # 추론에는 안 쓰임 (역직렬화 대체용)
        "clip_range": lambda _: 0.2,
    }
    model = PPO.load(a.model, env=env, custom_objects=custom_objects,
                     verbose=0, device="cpu")

    max_steps = int(a.max_time_s * a.control_hz)
    render_interval = max(a.control_hz // a.video_fps, 1)

    obs, _ = env.reset()
    frames = []
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if step % render_interval == 0:
            frames.append(env.render())
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    imageio.mimwrite(a.out, frames, fps=a.video_fps,
                     codec="libx264", quality=8, pixelformat="yuv420p")
    print("RENDER_DONE", a.out, f"({len(frames)} frames)")


if __name__ == "__main__":
    main()
