"""
PPO policy walking on a pre-generated terrain.

Usage:
    venv\Scripts\python view_ppo_terrain.py flat
    venv\Scripts\python view_ppo_terrain.py rough
    venv\Scripts\python view_ppo_terrain.py slope
    venv\Scripts\python view_ppo_terrain.py stairs
    venv\Scripts\python view_ppo_terrain.py hills
    venv\Scripts\python view_ppo_terrain.py random

Controls in viewer:
    SPACE – pause / unpause

Note: terrain is baked in at startup so it is fully visible.
      Re-run with a different argument to switch terrain.
"""

import os
import sys
import time
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO

from terrain_generator import TerrainGenerator

# ── args: terrain [--model <name>] ────────────────────────────────────────────
VALID   = ("flat", "rough", "slope", "stairs", "hills", "random")
args    = sys.argv[1:]
terrain = "flat"
model_name = "ppo_hexapod_flat"

for i, a in enumerate(args):
    if a == "--model" and i + 1 < len(args):
        model_name = args[i + 1]
    elif a.lower() in VALID:
        terrain = a.lower()

MODEL_PATH = model_name if model_name.endswith(".zip") else model_name + ".zip"
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: '{MODEL_PATH}' not found.")
    sys.exit(1)

# flat terrain → use the plain XML (same as training, no heightfield physics mismatch)
# non-flat     → use terrain XML with heightfield
_xml_file = "hexapod.xml" if terrain == "flat" else "hexapod_terrain.xml"
XML_PATH = os.path.join(
    os.path.dirname(__file__),
    "one_policy_to_run_them_all",
    "one_policy_to_run_them_all",
    "environments", "hexapod", "data", _xml_file,
)

NOMINAL = np.array([
    -0.7,  0.6, -2.0,
     0.7, -0.6,  2.0,
     0.0,  0.6, -2.0,
     0.0, -0.6,  2.0,
     0.7,  0.6, -2.0,
    -0.7, -0.6,  2.0,
])

DIFFICULTY = 1.0

# ── build model with terrain already set ──────────────────────────────────────
print(f"\nGenerating '{terrain}' terrain...")
m  = mujoco.MjModel.from_xml_path(XML_PATH)
d  = mujoco.MjData(m)

if terrain == "flat":
    spawn_z = 0.3
else:
    tg = TerrainGenerator(m)
    if terrain == "rough":
        tg.rough(difficulty=DIFFICULTY)
    elif terrain == "slope":
        tg.slope(difficulty=DIFFICULTY)
    elif terrain == "stairs":
        tg.stairs(difficulty=DIFFICULTY)
    elif terrain == "hills":
        tg.hills(difficulty=DIFFICULTY)
    elif terrain == "random":
        terrain = tg.random(difficulty=DIFFICULTY)
    spawn_z = tg.height_at(0.0, 0.0) + 0.25

# settle robot on top of terrain
mujoco.mj_resetData(m, d)
d.qpos[2]  = spawn_z
d.qpos[7:] = NOMINAL.copy()
mujoco.mj_forward(m, d)

for _ in range(500):
    err       = NOMINAL - d.qpos[7:]
    torques   = 20.0 * err - 0.8 * d.qvel[6:]
    d.ctrl[:] = np.clip(torques, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
    mujoco.mj_step(m, d)

print(f"Settled at height: {d.qpos[2]:.3f} m")

# ── load policy ───────────────────────────────────────────────────────────────
print(f"Loading PPO policy: {MODEL_PATH}")
policy = PPO.load(MODEL_PATH, device="cpu")

# ── helpers ───────────────────────────────────────────────────────────────────

def get_obs(d):
    joint_pos = (d.qpos[7:] - NOMINAL).astype(np.float32)
    joint_vel = d.qvel[6:].astype(np.float32)
    body_vel  = d.qvel[:3].astype(np.float32)
    quat      = d.qpos[3:7]
    euler     = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz").astype(np.float32)
    return np.concatenate([joint_pos, joint_vel, body_vel, euler])


def pd_control(m, d, action, kp=20.0, kd=0.8):
    target    = np.clip(NOMINAL + action,
                        m.actuator_ctrlrange[:, 0],
                        m.actuator_ctrlrange[:, 1])
    torques   = kp * (target - d.qpos[7:]) - kd * d.qvel[6:]
    d.ctrl[:] = np.clip(torques, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])


# ── viewer ────────────────────────────────────────────────────────────────────
paused = [False]

def key_cb(keycode):
    if keycode == ord(" "):
        paused[0] = not paused[0]

print(f"\nRunning PPO on '{terrain}' terrain  |  SPACE=pause  close window to quit\n")

ctrl_every = 10
action     = np.zeros(18)

# camera angle: low for terrains with height (slope/stairs), normal for flat/rough
cam_elevation = -5 if terrain in ("slope", "stairs", "hills") else -20

with mujoco.viewer.launch_passive(m, d, key_callback=key_cb) as viewer:
    viewer.cam.azimuth   = 135
    viewer.cam.elevation = cam_elevation
    viewer.cam.distance  = 5.0
    viewer.cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.3]

    step       = 0
    wall_start = time.time()
    start_x    = d.qpos[0]

    while viewer.is_running():
        if not paused[0]:
            if step % ctrl_every == 0:
                action, _ = policy.predict(get_obs(d), deterministic=True)

            pd_control(m, d, action)
            mujoco.mj_step(m, d)
            step += 1

        viewer.cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.3]
        viewer.sync()

        sim_time  = step * m.opt.timestep
        wall_time = time.time() - wall_start
        if sim_time > wall_time:
            time.sleep(sim_time - wall_time)

    dur  = step * m.opt.timestep
    dist = d.qpos[0] - start_x
    print(f"\n=== Session Summary ({terrain}) ===")
    print(f"  Distance : {dist:.3f} m")
    print(f"  Duration : {dur:.1f} s")
    print(f"  Avg speed: {dist/max(dur,1e-6):.3f} m/s")
    print(f"  Height   : {d.qpos[2]:.3f} m")
