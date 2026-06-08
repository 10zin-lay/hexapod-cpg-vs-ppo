"""
Evaluation comparison: CPG Tripod vs PPO on flat terrain.

Runs both controllers headless (no viewer) for 10 m and computes:
  - Speed, time-to-10m
  - Max / RMS Y deviation
  - Path efficiency (straight-line / actual path length)
  - Heading error at finish
  - Body height mean & stability
  - Energy consumed

Run:
    venv\Scripts\python compare_cpg_ppo.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO

# ─────────────────────────────────────────────────────────────────────────────
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

TARGET_DISTANCE = 10.0
MAX_SIM_STEPS   = 500_000   # safety cap (~1000 s sim time)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def pd_torques(m, d, target, kp=20.0, kd=0.8):
    err     = target - d.qpos[7:]
    torques = kp * err - kd * d.qvel[6:]
    return np.clip(torques, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])


def settle(m, d, kp=20.0, kd=0.8, steps=500):
    for _ in range(steps):
        d.ctrl[:] = pd_torques(m, d, NOMINAL, kp=kp, kd=kd)
        mujoco.mj_step(m, d)


def compute_metrics(x_pos, y_pos, z_pos, energies, start_x, start_y, dt):
    xs = np.array(x_pos)
    ys = np.array(y_pos)
    zs = np.array(z_pos)

    dist      = xs[-1] - start_x
    duration  = len(xs) * dt
    speed     = dist / duration if duration > 0 else 0.0

    y_dev     = ys - start_y
    max_y_dev = float(np.max(np.abs(y_dev)))
    rms_y_dev = float(np.sqrt(np.mean(y_dev ** 2)))

    # path length (sum of step-to-step distances)
    dx          = np.diff(xs)
    dy          = np.diff(ys)
    path_len    = float(np.sum(np.sqrt(dx**2 + dy**2)))
    efficiency  = dist / path_len if path_len > 0 else 1.0

    final_y_dev     = float(ys[-1] - start_y)
    heading_err_deg = float(np.degrees(np.arctan2(final_y_dev, dist)))

    return {
        "distance_m"      : float(dist),
        "duration_s"      : float(duration),
        "speed_ms"        : float(speed),
        "max_y_dev_m"     : max_y_dev,
        "rms_y_dev_m"     : rms_y_dev,
        "path_efficiency" : efficiency,
        "heading_err_deg" : heading_err_deg,
        "avg_height_m"    : float(np.mean(zs)),
        "height_std_m"    : float(np.std(zs)),
        "energy"          : float(np.sum(energies)),
        "fell_over"       : bool(np.min(zs) < 0.08),
        "x_pos"           : list(xs),
        "y_pos"           : list(ys),
        "z_pos"           : list(zs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CPG CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────

class HexapodCPG:
    def __init__(self, freq=1.0, hip_amp=0.4, knee_amp=0.3, ankle_amp=0.2):
        self.freq       = freq
        self.hip_amp    = hip_amp
        self.knee_amp   = knee_amp
        self.ankle_amp  = ankle_amp
        self.phase_offsets = np.array([0, np.pi, np.pi, 0, 0, np.pi])  # tripod

    def get_target(self, t):
        target     = NOMINAL.copy()
        hip_sign   = np.array([-1,  1, -1,  1, -1,  1])
        knee_sign  = np.array([ 1, -1,  1, -1,  1, -1])
        ankle_sign = np.array([ 1, -1,  1, -1,  1, -1])
        for leg in range(6):
            phase       = 2 * np.pi * self.freq * t + self.phase_offsets[leg]
            idx         = leg * 3
            target[idx]   += hip_sign[leg]   * self.hip_amp   * np.sin(phase)
            target[idx+1] += knee_sign[leg]  * self.knee_amp  * max(0.0, np.sin(phase))
            target[idx+2] += ankle_sign[leg] * self.ankle_amp * np.sin(phase + np.pi)
        return target


def run_cpg(freq=1.0, kp=20.0, kd=0.8):
    print("Running CPG (headless)...")
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2]  = 0.3
    d.qpos[7:] = NOMINAL.copy()
    mujoco.mj_forward(m, d)
    settle(m, d, kp=kp, kd=kd)

    cpg        = HexapodCPG(freq=freq)
    dt         = m.opt.timestep
    ctrl_every = 10

    start_x, start_y = d.qpos[0], d.qpos[1]
    x_pos, y_pos, z_pos, energies = [], [], [], []
    t = 0.0

    for step in range(MAX_SIM_STEPS):
        if d.qpos[0] - start_x >= TARGET_DISTANCE:
            break
        if step % ctrl_every == 0:
            target    = cpg.get_target(t)
            d.ctrl[:] = pd_torques(m, d, target, kp=kp, kd=kd)
        mujoco.mj_step(m, d)
        t += dt
        x_pos.append(d.qpos[0])
        y_pos.append(d.qpos[1])
        z_pos.append(d.qpos[2])
        energies.append(float(np.sum(np.abs(d.ctrl) * dt)))

    print(f"  CPG done — {d.qpos[0] - start_x:.2f} m in {len(x_pos)*dt:.1f} s")
    return compute_metrics(x_pos, y_pos, z_pos, energies, start_x, start_y, dt)


# ─────────────────────────────────────────────────────────────────────────────
# PPO CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────

def get_obs(m, d):
    joint_pos = (d.qpos[7:] - NOMINAL).astype(np.float32)
    joint_vel = d.qvel[6:].astype(np.float32)
    body_vel  = d.qvel[:3].astype(np.float32)
    quat      = d.qpos[3:7]
    euler     = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz").astype(np.float32)
    return np.concatenate([joint_pos, joint_vel, body_vel, euler])


def run_ppo(terrain="flat", kp=20.0, kd=0.8):
    model_path = f"ppo_hexapod_{terrain}.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Train first.")

    print(f"Running PPO [{terrain}] (headless)...")
    policy = PPO.load(model_path, device="cpu")

    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2]  = 0.3
    d.qpos[7:] = NOMINAL.copy()
    mujoco.mj_forward(m, d)
    settle(m, d, kp=kp, kd=kd)

    dt         = m.opt.timestep
    ctrl_every = 10
    start_x, start_y = d.qpos[0], d.qpos[1]
    x_pos, y_pos, z_pos, energies = [], [], [], []
    action = np.zeros(18)

    for step in range(MAX_SIM_STEPS):
        if d.qpos[0] - start_x >= TARGET_DISTANCE:
            break
        if step % ctrl_every == 0:
            obs, _  = get_obs(m, d), None
            action, _ = policy.predict(get_obs(m, d), deterministic=True)
            target    = np.clip(NOMINAL + action,
                                m.actuator_ctrlrange[:, 0],
                                m.actuator_ctrlrange[:, 1])
            torques   = kp * (target - d.qpos[7:]) - kd * d.qvel[6:]
            d.ctrl[:] = np.clip(torques, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        mujoco.mj_step(m, d)
        x_pos.append(d.qpos[0])
        y_pos.append(d.qpos[1])
        z_pos.append(d.qpos[2])
        energies.append(float(np.sum(np.abs(d.ctrl) * dt)))

    print(f"  PPO done — {d.qpos[0] - start_x:.2f} m in {len(x_pos)*dt:.1f} s")
    return compute_metrics(x_pos, y_pos, z_pos, energies, start_x, start_y, dt)


# ─────────────────────────────────────────────────────────────────────────────
# PRINT TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_table(cpg_m, ppo_m):
    metrics = [
        ("Speed (m/s)",          "speed_ms",        "{:.3f}",  True),
        ("Time to 10 m (s)",     "duration_s",      "{:.2f}",  False),
        ("Max Y deviation (m)",  "max_y_dev_m",     "{:.4f}",  False),
        ("RMS Y deviation (m)",  "rms_y_dev_m",     "{:.4f}",  False),
        ("Path efficiency",      "path_efficiency", "{:.4f}",  True),
        ("Heading error (°)",    "heading_err_deg", "{:+.2f}", False),
        ("Avg height (m)",       "avg_height_m",    "{:.3f}",  True),
        ("Height stability (σ)", "height_std_m",    "{:.4f}",  False),
        ("Energy (J·step)",      "energy",          "{:.2f}",  False),
        ("Fell over",            "fell_over",       "{}",      False),
    ]

    print("\n" + "="*65)
    print(f"  {'Metric':<26} {'CPG Tripod':>14}  {'PPO Flat':>14}  Winner")
    print("="*65)

    for label, key, fmt, higher_is_better in metrics:
        cv = cpg_m[key]
        pv = ppo_m[key]
        cs = fmt.format(cv)
        ps = fmt.format(pv)

        if key == "fell_over":
            winner = "CPG" if not cv and pv else ("PPO" if cv and not pv else "Tie")
        elif isinstance(cv, float) and isinstance(pv, float):
            if higher_is_better:
                winner = "PPO" if pv > cv else ("CPG" if cv > pv else "Tie")
            else:
                winner = "PPO" if pv < cv else ("CPG" if cv < pv else "Tie")
        else:
            winner = "—"

        print(f"  {label:<26} {cs:>14}  {ps:>14}  {winner}")

    print("="*65)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(cpg_m, ppo_m):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("CPG Tripod vs PPO — Flat Terrain (10 m run)", fontsize=14, fontweight="bold")

    colors = {"CPG": "#2196F3", "PPO": "#FF5722"}

    # ── 1. XY trajectory ──────────────────────────────────────────────────
    ax = axes[0, 0]
    cpg_xs = np.array(cpg_m["x_pos"]) - cpg_m["x_pos"][0]
    cpg_ys = np.array(cpg_m["y_pos"]) - cpg_m["y_pos"][0]
    ppo_xs = np.array(ppo_m["x_pos"]) - ppo_m["x_pos"][0]
    ppo_ys = np.array(ppo_m["y_pos"]) - ppo_m["y_pos"][0]
    ax.plot(cpg_xs, cpg_ys, color=colors["CPG"], linewidth=1.5, label="CPG")
    ax.plot(ppo_xs, ppo_ys, color=colors["PPO"], linewidth=1.5, label="PPO")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Ideal path")
    ax.set_xlabel("Forward distance (m)")
    ax.set_ylabel("Lateral deviation (m)")
    ax.set_title("XY Trajectory (top view)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # ── 2. Forward distance vs time ────────────────────────────────────────
    ax = axes[0, 1]
    cpg_dt  = cpg_m["duration_s"] / len(cpg_m["x_pos"])
    ppo_dt  = ppo_m["duration_s"] / len(ppo_m["x_pos"])
    cpg_t   = np.arange(len(cpg_m["x_pos"])) * cpg_dt
    ppo_t   = np.arange(len(ppo_m["x_pos"])) * ppo_dt
    ax.plot(cpg_t, cpg_xs, color=colors["CPG"], linewidth=1.5, label=f"CPG  {cpg_m['speed_ms']:.3f} m/s")
    ax.plot(ppo_t, ppo_xs, color=colors["PPO"], linewidth=1.5, label=f"PPO  {ppo_m['speed_ms']:.3f} m/s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Forward distance (m)")
    ax.set_title("Forward Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 3. Y deviation over distance ───────────────────────────────────────
    ax = axes[0, 2]
    ax.plot(cpg_xs, cpg_ys, color=colors["CPG"], linewidth=1.5, label="CPG")
    ax.plot(ppo_xs, ppo_ys, color=colors["PPO"], linewidth=1.5, label="PPO")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Forward distance (m)")
    ax.set_ylabel("Y deviation (m)")
    ax.set_title("Lateral Deviation over Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 4. Body height ─────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(cpg_xs, cpg_m["z_pos"], color=colors["CPG"], linewidth=1.2, label="CPG")
    ax.plot(ppo_xs, ppo_m["z_pos"], color=colors["PPO"], linewidth=1.2, label="PPO")
    ax.axhline(0.08, color="red", linestyle="--", linewidth=0.8, label="Fall threshold")
    ax.set_xlabel("Forward distance (m)")
    ax.set_ylabel("Body height (m)")
    ax.set_title("Body Height (Stability)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 5. Bar chart — key metrics ─────────────────────────────────────────
    ax = axes[1, 1]
    bar_metrics  = ["Speed\n(m/s)", "Max Y dev\n(m)", "Path\nefficiency"]
    cpg_vals     = [cpg_m["speed_ms"], cpg_m["max_y_dev_m"], cpg_m["path_efficiency"]]
    ppo_vals     = [ppo_m["speed_ms"], ppo_m["max_y_dev_m"], ppo_m["path_efficiency"]]
    x            = np.arange(len(bar_metrics))
    width        = 0.35
    bars1 = ax.bar(x - width/2, cpg_vals, width, label="CPG", color=colors["CPG"], alpha=0.85)
    bars2 = ax.bar(x + width/2, ppo_vals, width, label="PPO", color=colors["PPO"], alpha=0.85)
    ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_metrics)
    ax.set_title("Key Metrics Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # ── 6. Radar chart — normalised scores ────────────────────────────────
    ax = axes[1, 2]
    categories = ["Speed", "Straightness\n(inv Y-dev)", "Path\neff.", "Height\nstab.", "Energy\n(inv)"]
    N = len(categories)

    # normalise: higher = better for all (invert deviation and energy)
    max_speed = max(cpg_m["speed_ms"], ppo_m["speed_ms"])
    max_ydev  = max(cpg_m["max_y_dev_m"], ppo_m["max_y_dev_m"])
    max_e     = max(cpg_m["energy"], ppo_m["energy"])

    def norm_scores(m):
        return [
            m["speed_ms"]        / max_speed,
            1 - m["max_y_dev_m"] / max_ydev,
            m["path_efficiency"],
            1 - min(m["height_std_m"] / 0.05, 1.0),
            1 - m["energy"]      / max_e,
        ]

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for label, m, color in [("CPG", cpg_m, colors["CPG"]), ("PPO", ppo_m, colors["PPO"])]:
        scores = norm_scores(m) + [norm_scores(m)[0]]
        ax.plot(angles, scores, color=color, linewidth=2, label=label)
        ax.fill(angles, scores, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=8)
    ax.set_ylim(0, 1)
    ax.set_title("Normalised Performance\n(higher = better)")
    ax.legend(loc="upper right")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("comparison_cpg_ppo.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved → comparison_cpg_ppo.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cpg_metrics = run_cpg(freq=1.0, kp=20.0, kd=0.8)
    ppo_metrics = run_ppo(terrain="flat", kp=20.0, kd=0.8)

    print_table(cpg_metrics, ppo_metrics)
    plot_comparison(cpg_metrics, ppo_metrics)
