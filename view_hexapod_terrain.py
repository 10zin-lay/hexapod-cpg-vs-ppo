"""
Interactive terrain viewer for the hexapod (standing only, no walking).

Usage:
    venv\Scripts\python view_hexapod_terrain.py flat
    venv\Scripts\python view_hexapod_terrain.py rough
    venv\Scripts\python view_hexapod_terrain.py slope
    venv\Scripts\python view_hexapod_terrain.py stairs
    venv\Scripts\python view_hexapod_terrain.py hills
    venv\Scripts\python view_hexapod_terrain.py random

Controls:
    SPACE – pause / unpause
"""

import os
import sys
import time
import mujoco
import mujoco.viewer
import numpy as np

from terrain_generator import TerrainGenerator

XML_PATH = os.path.join(
    os.path.dirname(__file__),
    "one_policy_to_run_them_all",
    "one_policy_to_run_them_all",
    "environments", "hexapod", "data", "hexapod_terrain.xml",
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

# ── pick terrain from argument ─────────────────────────────────────────────────
VALID   = ("flat", "rough", "slope", "stairs", "hills", "random")
terrain = sys.argv[1].lower() if len(sys.argv) > 1 else "flat"
if terrain not in VALID:
    print(f"Unknown terrain '{terrain}'. Choose from: {', '.join(VALID)}")
    sys.exit(1)

# ── build model with terrain baked in before viewer opens ──────────────────────
print(f"\nGenerating '{terrain}' terrain...")
m  = mujoco.MjModel.from_xml_path(XML_PATH)
d  = mujoco.MjData(m)
tg = TerrainGenerator(m)

if terrain == "flat":
    tg.flat()
elif terrain == "rough":
    tg.rough(difficulty=DIFFICULTY)
elif terrain == "slope":
    tg.slope(difficulty=DIFFICULTY)
elif terrain == "stairs":
    tg.stairs(difficulty=DIFFICULTY)
elif terrain == "hills":
    tg.hills(difficulty=DIFFICULTY)
elif terrain == "random":
    terrain = tg.random(difficulty=DIFFICULTY)

# settle robot on terrain
mujoco.mj_resetData(m, d)
d.qpos[2]  = tg.height_at(0.0, 0.0) + 0.25
d.qpos[7:] = NOMINAL.copy()
mujoco.mj_forward(m, d)

for _ in range(400):
    err       = NOMINAL - d.qpos[7:]
    torques   = 20.0 * err - 0.8 * d.qvel[6:]
    d.ctrl[:] = np.clip(torques, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
    mujoco.mj_step(m, d)

print(f"Settled at height: {d.qpos[2]:.3f} m")
print(f"Terrain: {terrain}  |  SPACE=pause  close window to quit\n")

# ── viewer ─────────────────────────────────────────────────────────────────────
paused = [False]

def key_cb(keycode):
    if keycode == ord(" "):
        paused[0] = not paused[0]

cam_elevation = -5 if terrain in ("slope", "stairs", "hills") else -20

with mujoco.viewer.launch_passive(m, d, key_callback=key_cb) as viewer:
    viewer.cam.azimuth   = 135
    viewer.cam.elevation = cam_elevation
    viewer.cam.distance  = 5.0
    viewer.cam.lookat[:] = [0.0, 0.0, 0.3]

    step       = 0
    wall_start = time.time()

    while viewer.is_running():
        if not paused[0]:
            err       = NOMINAL - d.qpos[7:]
            torques   = 20.0 * err - 0.8 * d.qvel[6:]
            d.ctrl[:] = np.clip(torques, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
            mujoco.mj_step(m, d)
            step += 1

        viewer.cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.3]
        viewer.sync()

        sim_time  = step * m.opt.timestep
        wall_time = time.time() - wall_start
        if sim_time > wall_time:
            time.sleep(sim_time - wall_time)


if __name__ == "__main__":
    pass
