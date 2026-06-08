"""
View a trained PPO policy in the MuJoCo interactive viewer.

Usage:
    venv\Scripts\python view_ppo.py                  # default: flat terrain
    venv\Scripts\python view_ppo.py rough             # rough terrain model
    venv\Scripts\python view_ppo.py slope             # slope terrain model

Controls in viewer:
    Left-drag  → rotate
    Right-drag → pan
    Scroll     → zoom
    SPACE      → pause / unpause
"""

import os
import sys
import time
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO

# ── pick terrain from command line arg (default: flat) ──────────────────────
terrain = sys.argv[1] if len(sys.argv) > 1 else "flat"
model_path = f"ppo_hexapod_{terrain}.zip"

if not os.path.exists(model_path):
    print(f"ERROR: model file '{model_path}' not found.")
    print("Train it first with:  venv\\Scripts\\python train_hexapod_ppo.py")
    sys.exit(1)

XML_PATH = os.path.join(
    os.path.dirname(__file__),
    "one_policy_to_run_them_all",
    "one_policy_to_run_them_all",
    "environments", "hexapod", "data", "hexapod.xml",
)

NOMINAL = np.array([
    -0.7,  0.6, -2.0,
     0.7, -0.6,  2.0,
     0.0,  0.6, -2.0,
     0.0, -0.6,  2.0,
     0.7,  0.6, -2.0,
    -0.7, -0.6,  2.0,
])


def get_obs(m, d):
    joint_pos = (d.qpos[7:] - NOMINAL).astype(np.float32)
    joint_vel = d.qvel[6:].astype(np.float32)
    body_vel  = d.qvel[:3].astype(np.float32)
    quat      = d.qpos[3:7]
    euler     = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz").astype(np.float32)
    return np.concatenate([joint_pos, joint_vel, body_vel, euler])


def settle(m, d, kp=20.0, kd=0.8, steps=300):
    for _ in range(steps):
        err       = NOMINAL - d.qpos[7:]
        torques   = kp * err - kd * d.qvel[6:]
        d.ctrl[:] = np.clip(torques, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        mujoco.mj_step(m, d)


def pd_control(m, d, action, kp=20.0, kd=0.8):
    target    = np.clip(NOMINAL + action,
                        m.actuator_ctrlrange[:, 0],
                        m.actuator_ctrlrange[:, 1])
    torques   = kp * (target - d.qpos[7:]) - kd * d.qvel[6:]
    d.ctrl[:] = np.clip(torques, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])


# ── load model ───────────────────────────────────────────────────────────────
print(f"\nLoading PPO model: {model_path}")
policy = PPO.load(model_path, device="cpu")

m = mujoco.MjModel.from_xml_path(XML_PATH)
d = mujoco.MjData(m)

# ── reset & settle ───────────────────────────────────────────────────────────
mujoco.mj_resetData(m, d)
d.qpos[2]  = 0.3
d.qpos[7:] = NOMINAL.copy()
mujoco.mj_forward(m, d)
settle(m, d)

print(f"Settled at height: {d.qpos[2]:.3f} m")
print(f"Running '{terrain}' policy — close the window to quit.\n")

ctrl_every = 10   # policy runs at 50 Hz (same as training)
paused     = [False]

def key_cb(keycode):
    if keycode == ord(" "):
        paused[0] = not paused[0]

TARGET_DISTANCE = 10.0  # stop after this many metres (matches CPG deviation test)

print(f"\n  {'X (m)':<10} {'Y deviation (m)':<20} {'Heading error (°)':<20} {'Height (m)'}")
print(f"  {'-'*65}")

with mujoco.viewer.launch_passive(m, d, key_callback=key_cb) as viewer:
    viewer.cam.azimuth   = 90
    viewer.cam.elevation = -20
    viewer.cam.distance  = 2.5
    viewer.cam.lookat[:] = [0.0, 0.0, 0.2]

    step          = 0
    action        = np.zeros(18)
    wall_start    = time.time()
    start_x       = d.qpos[0]
    start_y       = d.qpos[1]
    next_debug_x  = 1.0

    y_positions   = []
    z_heights     = []

    while viewer.is_running():
        forward_dist = d.qpos[0] - start_x
        if forward_dist >= TARGET_DISTANCE:
            print(f"\n  Reached {TARGET_DISTANCE}m — stopping.")
            break

        if not paused[0]:
            if step % ctrl_every == 0:
                obs       = get_obs(m, d)
                action, _ = policy.predict(obs, deterministic=True)

            pd_control(m, d, action)
            mujoco.mj_step(m, d)
            step += 1

            y_positions.append(d.qpos[1])
            z_heights.append(d.qpos[2])

            # debug print every 1 metre
            if forward_dist >= next_debug_x:
                y_dev       = d.qpos[1] - start_y
                heading_err = np.degrees(np.arctan2(y_dev, forward_dist))
                print(f"  {forward_dist:<10.2f} {y_dev:<+20.4f} {heading_err:<+20.2f} {d.qpos[2]:.3f}")
                next_debug_x += 1.0

            if step % 10 == 0:
                viewer.cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.2]

        viewer.sync()

        sim_time  = step * m.opt.timestep
        wall_time = time.time() - wall_start
        if sim_time > wall_time:
            time.sleep(sim_time - wall_time)

    dur  = step * m.opt.timestep
    dist = d.qpos[0] - start_x
    max_y_dev = float(np.max(np.abs(np.array(y_positions) - start_y))) if y_positions else 0.0

    print(f"\n=== PPO Session Summary ===")
    print(f"  Distance      : {dist:.3f} m")
    print(f"  Duration      : {dur:.1f} s")
    print(f"  Avg speed     : {dist/max(dur,1e-6):.3f} m/s")
    print(f"  Max Y deviation: {max_y_dev:.4f} m")
    print(f"  Avg height    : {np.mean(z_heights):.3f} m  (fell if < 0.08)")
