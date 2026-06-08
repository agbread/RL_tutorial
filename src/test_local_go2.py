import time
import yaml
from pathlib import Path

import mujoco
import imageio
from tqdm.auto import tqdm

if not hasattr(mujoco.MjData, 'solver_iter'):
    setattr(mujoco.MjData, 'solver_iter', property(lambda self: self.solver_niter))

from stable_baselines3 import PPO
from go2_mujoco_env import Go2MujocoEnv


def test():
    base_dir = Path(__file__).resolve().parents[1]

    model_name = "2026-03-19_16-27-15"
    model_path = base_dir / "models" / model_name / "best_model.zip"
    cfg_path = base_dir / "src" / "params.yaml"

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    given_command = [0.9, 0.0, 0.0]

    env = Go2MujocoEnv(
        prj_path=base_dir.as_posix(),
        given_command=given_command,
        render_mode="rgb_array",
        camera_name="tracking",
        width=2560,
        height=1440,
    )

    t_render = 0.0
    n_render = 0
    last_render = 0.0
    video_path = base_dir / "models" / model_name / f"rollout_{model_name}.mp4"

    try:
        model = PPO.load(path=model_path, env=env, verbose=1)

        max_time_step_s = cfg["test"]["max_time_step_s"]
        video_fps = 10

        # control rate = 50 Hz
        render_interval = 50 // video_fps
        max_steps = int(max_time_step_s * 50)   # 50 Hz면 이게 더 자연스러움

        print("max time:", max_time_step_s)
        print("max steps:", max_steps)

        frames = []
        pbar = tqdm(total=max_steps, desc="rollout", unit="step", dynamic_ncols=True)

        obs, _ = env.reset()
        start = time.perf_counter()

        ep_len = 0
        ep_reward = 0.0
        total_reward = 0.0

        for global_step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_len += 1
            total_reward += reward

            time.sleep(0.02)

            if global_step % render_interval == 0:
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
                print(f"episode finished: ep_len={ep_len}, ep_reward={ep_reward:.3f}")
                obs, _ = env.reset()
                ep_len = 0
                ep_reward = 0.0

        imageio.mimwrite(
            video_path,
            frames,
            fps=video_fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        )

    finally:
        env.close()
        print("avg render sec:", t_render / max(n_render, 1))
        print("Saved video to:", video_path)


if __name__ == "__main__":
    test()