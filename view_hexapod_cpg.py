import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time

XML_PATH = (
    "one_policy_to_run_them_all/"
    "one_policy_to_run_them_all/"
    "environments/hexapod/data/hexapod.xml"
)

NOMINAL = np.array([
    -0.7,  0.6, -2.0,   # leg 0
     0.7, -0.6,  2.0,   # leg 1
     0.0,  0.6, -2.0,   # leg 2
     0.0, -0.6,  2.0,   # leg 3
     0.7,  0.6, -2.0,   # leg 4
    -0.7, -0.6,  2.0,   # leg 5
])


class HexapodCPG:
    def __init__(self, gait="tripod", freq=1.5,
                 hip_amp=0.4, knee_amp=0.3, ankle_amp=0.2):
        self.freq       = freq
        self.hip_amp    = hip_amp
        self.knee_amp   = knee_amp
        self.ankle_amp  = ankle_amp
        self.set_gait(gait)

    def set_gait(self, gait):
        self.gait = gait
        # Leg layout: 0=FL, 1=FR, 2=ML, 3=MR, 4=RL, 5=RR
        if gait == "tripod":
            # Group A (phase 0):  FL(0), MR(3), RL(4)  — diagonal triangle
            # Group B (phase π):  FR(1), ML(2), RR(5)  — other triangle
            self.phase_offsets = np.array([0, np.pi, np.pi, 0, 0, np.pi])
        elif gait == "wave":
            # Each leg offset by π/3 — one leg lifts at a time
            self.phase_offsets = np.array([
                0, np.pi, np.pi/3, 4*np.pi/3, 2*np.pi/3, 5*np.pi/3
            ])
        elif gait == "ripple":
            # Overlapping wave — two legs always in swing
            self.phase_offsets = np.array([
                0, np.pi, 2*np.pi/3, 5*np.pi/3, 4*np.pi/3, np.pi/3
            ])

    def get_target(self, t):
        target = NOMINAL.copy()
        # All three joints are mirrored left/right in the XML (see NOMINAL signs).
        # Each needs its own sign so the physical motion is symmetric.
        # Left(0,2,4)=-1  Right(1,3,5)=+1  for hip  and ankle (same direction)
        # Left(0,2,4)=+1  Right(1,3,5)=-1  for knee  (opposite: +0.6 vs -0.6 nominal)
        hip_sign   = np.array([-1,  1, -1,  1, -1,  1])
        knee_sign  = np.array([ 1, -1,  1, -1,  1, -1])
        ankle_sign = np.array([ 1, -1,  1, -1,  1, -1])

        for leg in range(6):
            phase = 2 * np.pi * self.freq * t + self.phase_offsets[leg]
            idx   = leg * 3
            target[idx + 0] += hip_sign[leg]   * self.hip_amp   * np.sin(phase)
            target[idx + 1] += knee_sign[leg]  * self.knee_amp  * max(0.0, np.sin(phase))
            target[idx + 2] += ankle_sign[leg] * self.ankle_amp * np.sin(phase + np.pi)
        return target


def pd_torques(m, d, target, kp=15.0, kd=0.5):
    pos_error = target - d.qpos[7:]
    vel       = d.qvel[6:]
    torques   = kp * pos_error - kd * vel
    return np.clip(torques, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])


def run_gait(gait="tripod", freq=1.0, target_distance=10.0, kp=20.0, kd=0.8,
             hip_amp=0.4, knee_amp=0.3, ankle_amp=0.2):
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    mujoco.mj_resetData(m, d)
    d.qpos[2]  = 0.3
    d.qpos[7:] = NOMINAL.copy()
    mujoco.mj_forward(m, d)

    cpg        = HexapodCPG(gait=gait, freq=freq,
                             hip_amp=hip_amp, knee_amp=knee_amp, ankle_amp=ankle_amp)
    dt         = m.opt.timestep
    ctrl_every = 10

    x_positions = []
    y_positions = []
    z_heights   = []
    energies    = []

    # print deviation every 1 metre of forward travel
    next_debug_x = 1.0

    print(f"\n  Opening viewer for '{gait}' gait — running until {target_distance}m forward.")
    print(f"  {'X (m)':<10} {'Y deviation (m)':<20} {'Heading error (°)':<20} {'Height (m)'}")
    print(f"  {'-'*65}")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.azimuth   = 90
        viewer.cam.elevation = -20
        viewer.cam.distance  = 2.5
        viewer.cam.lookat[:] = [0.0, 0.0, 0.2]

        # Phase 1 — settle
        for step in range(500):
            d.ctrl[:] = pd_torques(m, d, NOMINAL, kp=kp, kd=kd)
            mujoco.mj_step(m, d)
            if step % 10 == 0:
                viewer.sync()

        start_x = d.qpos[0]
        start_y = d.qpos[1]
        print(f"  Settled at height: {d.qpos[2]:.3f} m  |  start pos: ({start_x:.3f}, {start_y:.3f})")

        # Phase 2 — walk until target_distance reached
        t          = 0.0
        step       = 0
        wall_start = time.time()

        while viewer.is_running():
            forward_dist = d.qpos[0] - start_x
            if forward_dist >= target_distance:
                print(f"\n  Reached {target_distance}m — stopping.")
                break

            if step % ctrl_every == 0:
                target = cpg.get_target(t)
                d.ctrl[:] = pd_torques(m, d, target, kp=kp, kd=kd)

            mujoco.mj_step(m, d)
            t    += dt
            step += 1

            x_positions.append(d.qpos[0])
            y_positions.append(d.qpos[1])
            z_heights.append(d.qpos[2])
            energies.append(float(np.sum(np.abs(d.ctrl) * dt)))

            # debug print every 1 metre
            if forward_dist >= next_debug_x:
                y_dev        = d.qpos[1] - start_y
                heading_err  = np.degrees(np.arctan2(y_dev, forward_dist))
                print(f"  {forward_dist:<10.2f} {y_dev:<+20.4f} {heading_err:<+20.2f} {d.qpos[2]:.3f}")
                next_debug_x += 1.0

            if step % 10 == 0:
                viewer.cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.2]
                viewer.sync()

                sim_time  = step * dt
                wall_time = time.time() - wall_start
                if sim_time > wall_time:
                    time.sleep(sim_time - wall_time)

    duration = step * dt if step > 0 else 1.0
    metrics = {
        "distance_travelled": float(x_positions[-1] - x_positions[0]) if x_positions else 0.0,
        "avg_velocity"      : float((x_positions[-1] - x_positions[0]) / duration) if x_positions else 0.0,
        "avg_body_height"   : float(np.mean(z_heights)) if z_heights else 0.0,
        "stability"         : float(np.std(z_heights)) if z_heights else 0.0,
        "energy_consumed"   : float(np.sum(energies)),
        "fell_over"         : bool(np.min(z_heights) < 0.08) if z_heights else True,
        "max_y_deviation"   : float(np.max(np.abs(np.array(y_positions) - (y_positions[0] if y_positions else 0)))) if y_positions else 0.0,
        "x_positions"       : x_positions,
        "y_positions"       : y_positions,
        "z_heights"         : z_heights,
    }
    return metrics


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
gaits   = ["tripod"]  # , "wave", "ripple"]
results = {}

for gait in gaits:
    print(f"\nRunning {gait} gait...")
    metrics          = run_gait(gait=gait, freq=1.0, target_distance=10.0, kp=20.0, kd=0.8)
    results[gait]    = metrics
    print(f"  Distance:  {metrics['distance_travelled']:.3f} m")
    print(f"  Avg vel:   {metrics['avg_velocity']:.3f} m/s")
    print(f"  Height:    {metrics['avg_body_height']:.3f} m")
    print(f"  Fell over: {metrics['fell_over']}")


# ─────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax1 = axes[0]
for gait in gaits:
    x = results[gait]["x_positions"]
    t = np.linspace(0, 8.0, len(x))
    ax1.plot(t, x, label=gait.capitalize(), linewidth=2)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Forward Distance (m)")
ax1.set_title("Forward Progress")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
for gait in gaits:
    z = results[gait]["z_heights"]
    t = np.linspace(0, 8.0, len(z))
    ax2.plot(t, z, label=gait.capitalize(), linewidth=2)
ax2.axhline(y=0.08, color="red", linestyle="--", label="Fall threshold")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Body Height (m)")
ax2.set_title("Stability (Body Height)")
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = axes[2]
gait_names = [g.capitalize() for g in gaits]
distances  = [results[g]["distance_travelled"] for g in gaits]
colors     = ["#2196F3", "#4CAF50", "#FF9800"]
bars = ax3.bar(gait_names, distances, color=colors, alpha=0.8)
ax3.bar_label(bars, fmt="%.3f m", padding=3)
ax3.set_ylabel("Distance Travelled (m)")
ax3.set_title("Total Distance by Gait")
ax3.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("cpg_results.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n=== SUMMARY TABLE ===")
print(f"{'Gait':<10} {'Distance(m)':<14} {'Vel(m/s)':<12} {'Height(m)':<12} {'Stability':<12} {'Energy':<12} {'Fell?'}")
print("-" * 75)
for gait in gaits:
    mm = results[gait]
    print(f"{gait:<10} {mm['distance_travelled']:<14.3f} {mm['avg_velocity']:<12.3f} "
          f"{mm['avg_body_height']:<12.3f} {mm['stability']:<12.4f} "
          f"{mm['energy_consumed']:<12.3f} {mm['fell_over']}")
