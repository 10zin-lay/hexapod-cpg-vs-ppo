"""
Interactive terrain viewer for the hexapod.

Press keys while the viewer is open:
  1 – flat terrain
  2 – rough terrain
  3 – slope terrain
  4 – stairs terrain
  5 – hills terrain
  R – random terrain (re-randomised each press)
  SPACE – pause / unpause
"""

import os
import mujoco
import mujoco.viewer
import numpy as np
import time

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

DIFFICULTY = 0.6   # adjust 0–1


def reset_robot(m, d, tg):
    mujoco.mj_resetData(m, d)
    spawn_z    = tg.height_at(0.0, 0.0) + 0.22
    d.qpos[2]  = spawn_z
    d.qpos[7:] = NOMINAL.copy()
    mujoco.mj_forward(m, d)

    # settle into standing pose
    for _ in range(400):
        err            = NOMINAL - d.qpos[7:]
        torques        = 20.0 * err - 0.8 * d.qvel[6:]
        d.ctrl[:]      = np.clip(torques,
                                  m.actuator_ctrlrange[:, 0],
                                  m.actuator_ctrlrange[:, 1])
        mujoco.mj_step(m, d)


def apply_terrain(name, tg, m, d):
    print(f"\n[Viewer] switching to: {name}")
    if name == "flat":
        tg.flat()
    elif name == "rough":
        tg.rough(difficulty=DIFFICULTY)
    elif name == "slope":
        tg.slope(difficulty=DIFFICULTY)
    elif name == "stairs":
        tg.stairs(difficulty=DIFFICULTY)
    elif name == "hills":
        tg.hills(difficulty=DIFFICULTY)
    elif name == "random":
        tg.random(difficulty=DIFFICULTY)
    # notify MuJoCo to rebuild the collision mesh
    mujoco.mj_resetData(m, d)
    reset_robot(m, d, tg)


def main():
    m  = mujoco.MjModel.from_xml_path(XML_PATH)
    d  = mujoco.MjData(m)
    tg = TerrainGenerator(m)

    # start flat
    tg.flat()
    reset_robot(m, d, tg)

    pending_terrain = [None]   # list used so closure can mutate it
    paused          = [False]

    def key_cb(keycode):
        mapping = {
            ord("1"): "flat",
            ord("2"): "rough",
            ord("3"): "slope",
            ord("4"): "stairs",
            ord("5"): "hills",
            ord("R"): "random",
            ord("r"): "random",
        }
        if keycode in mapping:
            pending_terrain[0] = mapping[keycode]
        elif keycode == ord(" "):
            paused[0] = not paused[0]

    print("Controls: 1=flat  2=rough  3=slope  4=stairs  5=hills  R=random  SPACE=pause")

    with mujoco.viewer.launch_passive(m, d, key_callback=key_cb) as viewer:
        viewer.cam.azimuth   = 45
        viewer.cam.elevation = -20
        viewer.cam.distance  = 3.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.3]

        wall_start = time.time()
        step       = 0

        while viewer.is_running():
            # handle terrain switch requested by keypress
            if pending_terrain[0] is not None:
                apply_terrain(pending_terrain[0], tg, m, d)
                pending_terrain[0] = None
                wall_start = time.time()
                step       = 0

            if not paused[0]:
                # simple standing controller (PD on nominal)
                err       = NOMINAL - d.qpos[7:]
                torques   = 20.0 * err - 0.8 * d.qvel[6:]
                d.ctrl[:] = np.clip(torques,
                                    m.actuator_ctrlrange[:, 0],
                                    m.actuator_ctrlrange[:, 1])
                mujoco.mj_step(m, d)
                step += 1

            viewer.cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.3]
            viewer.sync()

            # real-time pacing
            sim_time  = step * m.opt.timestep
            wall_time = time.time() - wall_start
            if sim_time > wall_time:
                time.sleep(sim_time - wall_time)


if __name__ == "__main__":
    main()
