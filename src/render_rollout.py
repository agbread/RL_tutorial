
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

    def log(m):
        print("[RR]", m, file=sys.stderr, flush=True)

    os.environ["MUJOCO_GL"] = a.gl

    if a.gl in ("egl", "osmesa"):
        os.environ["PYOPENGL_PLATFORM"] = a.gl
    sys.path.insert(0, a.prj)
    sys.path.insert(0, os.path.join(a.prj, "src"))

    log(f"start gl={a.gl}")
    import torch  # noqa: F401  (mujoco 보다 먼저)
    log("torch imported")
    import imageio
    import mujoco
    from stable_baselines3 import PPO
    log("sb3/mujoco imported")
    import src.go2_mujoco_env as go2_env
    log("go2 env module imported")

    env = go2_env.Go2MujocoEnv(
        prj_path=a.prj, cfg_path=a.cfg, given_command=a.command, render_mode=None,
    )
    env._reset_noise_scale = 0.05
    log("env created")

    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "lr_schedule": lambda _: 1e-4,
        "clip_range": lambda _: 0.2,
    }
    model = PPO.load(a.model, env=env, custom_objects=custom_objects,
                     verbose=0, device="cpu")
    log("model loaded (cpu)")

    # mujoco.Renderer (gymnasium OffScreenViewer 대신)
    renderer = mujoco.Renderer(env.model, height=a.height, width=a.width)
    log("mujoco.Renderer created")

    max_steps = int(a.max_time_s * a.control_hz)
    render_interval = max(a.control_hz // a.video_fps, 1)

    obs, _ = env.reset()
    frames = []
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if step % render_interval == 0:
            renderer.update_scene(env.data, camera=a.camera)
            frames.append(renderer.render().copy())
            if step == 0:
                log("first render OK")
        if terminated or truncated:
            obs, _ = env.reset()
    renderer.close()
    env.close()
    log(f"rollout done ({len(frames)} frames)")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    imageio.mimwrite(a.out, frames, fps=a.video_fps,
                     codec="libx264", quality=8, pixelformat="yuv420p")
    print("RENDER_DONE", a.out, f"({len(frames)} frames)")


if __name__ == "__main__":
    main()
