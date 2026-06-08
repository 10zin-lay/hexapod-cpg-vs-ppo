"""
Hexapod PPO training — VSCode / local GPU version.

Before running, make sure you have the CUDA build of PyTorch:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install stable-baselines3 scipy

Run:
    python train_hexapod_ppo.py
"""

import os
import pickle

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import mujoco
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
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

# SB3's PPO with MlpPolicy runs faster on CPU — GPU helps CNN policies only
DEVICE = "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
class HexapodEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, terrain="flat"):
        super().__init__()
        self.terrain    = terrain
        self.m          = mujoco.MjModel.from_xml_path(XML_PATH)
        self.d          = mujoco.MjData(self.m)
        self.dt         = self.m.opt.timestep
        self.ctrl_every = 10
        self.max_steps  = 1000
        self.step_count = 0
        self.kp         = 20.0
        self.kd         = 0.8

        # obs: 18 joint pos + 18 joint vel + 3 body vel + 3 euler = 42
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(42,), dtype=np.float32
        )
        # action: target joint offsets from nominal
        self.action_space = spaces.Box(
            low=-0.5, high=0.5, shape=(18,), dtype=np.float32
        )

    def _get_obs(self):
        joint_pos = (self.d.qpos[7:] - NOMINAL).astype(np.float32)
        joint_vel = self.d.qvel[6:].astype(np.float32)
        body_vel  = self.d.qvel[:3].astype(np.float32)
        quat      = self.d.qpos[3:7]
        euler     = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz").astype(np.float32)
        return np.concatenate([joint_pos, joint_vel, body_vel, euler])

    def _apply_terrain(self):
        if self.terrain == "rough":
            if self.step_count % 50 == 0:
                self.d.qfrc_applied[:3] = np.random.uniform(-0.5, 0.5, 3)
            else:
                self.d.qfrc_applied[:3] = 0.0
        elif self.terrain == "slope":
            self.m.opt.gravity[0] =  2.0
            self.m.opt.gravity[2] = -9.61

    def _settle(self):
        for _ in range(300):
            err            = NOMINAL - self.d.qpos[7:]
            torques        = self.kp * err - self.kd * self.d.qvel[6:]
            self.d.ctrl[:] = np.clip(torques,
                                     self.m.actuator_ctrlrange[:, 0],
                                     self.m.actuator_ctrlrange[:, 1])
            mujoco.mj_step(self.m, self.d)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # reset gravity if slope was used
        self.m.opt.gravity[:] = [0.0, 0.0, -9.81]
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[2]  = 0.3
        self.d.qpos[7:] = NOMINAL.copy()
        mujoco.mj_forward(self.m, self.d)
        self._settle()
        self.step_count = 0
        self.prev_x     = self.d.qpos[0]
        return self._get_obs(), {}

    def step(self, action):
        self._apply_terrain()

        target         = np.clip(NOMINAL + action,
                                 self.m.actuator_ctrlrange[:, 0],
                                 self.m.actuator_ctrlrange[:, 1])
        torques        = self.kp * (target - self.d.qpos[7:]) - self.kd * self.d.qvel[6:]
        self.d.ctrl[:] = np.clip(torques,
                                  self.m.actuator_ctrlrange[:, 0],
                                  self.m.actuator_ctrlrange[:, 1])

        for _ in range(self.ctrl_every):
            mujoco.mj_step(self.m, self.d)

        self.step_count += 1
        obs    = self._get_obs()
        height = self.d.qpos[2]
        quat   = self.d.qpos[3:7]
        euler  = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz")
        tilt   = abs(euler[0]) + abs(euler[1])

        forward_vel_ms = (self.d.qpos[0] - self.prev_x) / (self.ctrl_every * self.dt)
        lateral_vel_ms = abs(self.d.qvel[1])
        self.prev_x    = self.d.qpos[0]

        reward = (
             8.0 * forward_vel_ms
            - 2.0 * lateral_vel_ms
            - 2.0 * tilt
            - 0.001 * float(np.sum(action ** 2))
        )

        terminated = bool(height < 0.08 or tilt > 1.0)
        truncated  = bool(self.step_count >= self.max_steps)
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK — tracks episode rewards
# ─────────────────────────────────────────────────────────────────────────────
class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._ep_reward      = 0.0

    def _on_step(self):
        self._ep_reward += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            self._ep_reward = 0.0
        return True


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────
def _latest_checkpoint(ckpt_dir: str, prefix: str):
    """Return the path of the highest-step checkpoint, or None."""
    if not os.path.isdir(ckpt_dir):
        return None
    files = [
        f for f in os.listdir(ckpt_dir)
        if f.startswith(prefix) and f.endswith(".zip")
    ]
    if not files:
        return None
    # filenames: <prefix>_<steps>_steps.zip
    def step_count(name):
        try:
            return int(name.replace(prefix + "_", "").replace("_steps.zip", ""))
        except ValueError:
            return -1
    best = max(files, key=step_count)
    return os.path.join(ckpt_dir, best), step_count(best)


def train_terrain(terrain: str, total_timesteps: int = 500_000):
    print(f"\n{'='*55}")
    print(f"  Training on '{terrain}' terrain  |  device: {DEVICE}")
    print(f"{'='*55}")

    ckpt_dir   = os.path.join("checkpoints", terrain)
    ckpt_prefix = f"ppo_hexapod_{terrain}"
    os.makedirs(ckpt_dir, exist_ok=True)

    env           = DummyVecEnv([lambda t=terrain: HexapodEnv(terrain=t)])
    reward_logger = RewardLogger()
    checkpoint_cb = CheckpointCallback(
        save_freq       = 50_000,
        save_path       = ckpt_dir,
        name_prefix     = ckpt_prefix,
        verbose         = 1,
    )
    callbacks = CallbackList([reward_logger, checkpoint_cb])

    # ── resume from latest checkpoint if one exists ──
    resume = _latest_checkpoint(ckpt_dir, ckpt_prefix)
    if resume:
        ckpt_path, steps_done = resume
        remaining = total_timesteps - steps_done
        if remaining <= 0:
            print(f"  Already trained {steps_done:,} steps — skipping.")
            env.close()
            return []
        print(f"  Resuming from checkpoint: {ckpt_path}  ({steps_done:,} steps done)")
        model = PPO.load(ckpt_path, env=env, device=DEVICE,
                         tensorboard_log=f"./logs/{terrain}")
    else:
        remaining = total_timesteps
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate   = 3e-4,
            n_steps         = 4096,
            batch_size      = 256,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.01,
            verbose         = 1,
            device          = DEVICE,
            tensorboard_log = f"./logs/{terrain}",
        )

    model.learn(
        total_timesteps      = remaining,
        callback             = callbacks,
        progress_bar         = True,
        reset_num_timesteps  = resume is None,  # keep step count when resuming
    )

    final_path = f"ppo_hexapod_{terrain}"
    model.save(final_path)
    print(f"  Final model saved → {final_path}.zip")
    if reward_logger.episode_rewards:
        print(f"  Final avg reward (last 20 eps): "
              f"{np.mean(reward_logger.episode_rewards[-20:]):.2f}")

    env.close()
    return reward_logger.episode_rewards


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE A SAVED MODEL
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(terrain: str, n_episodes: int = 5):
    model = PPO.load(f"ppo_hexapod_{terrain}", device=DEVICE)
    env   = HexapodEnv(terrain=terrain)

    ep_rewards, ep_distances = [], []

    for ep in range(n_episodes):
        obs, _     = env.reset()
        done       = False
        total_r    = 0.0
        start_x    = env.d.qpos[0]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total_r  += r
            done      = terminated or truncated

        ep_rewards.append(total_r)
        ep_distances.append(env.d.qpos[0] - start_x)
        print(f"  [{terrain}] ep {ep+1}: reward={total_r:.1f}  dist={ep_distances[-1]:.3f}m")

    env.close()
    return ep_rewards, ep_distances


# ─────────────────────────────────────────────────────────────────────────────
# PLOT LEARNING CURVES
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(all_rewards: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors    = {"flat": "#2196F3", "rough": "#4CAF50", "slope": "#FF9800"}

    ax = axes[0]
    for terrain, rewards in all_rewards.items():
        smoothed = np.convolve(rewards, np.ones(20)/20, mode="valid")
        ax.plot(smoothed, label=terrain.capitalize(),
                color=colors[terrain], linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Learning Curves (20-ep rolling avg)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    terrains   = list(all_rewards.keys())
    final_avgs = [np.mean(all_rewards[t][-20:]) for t in terrains]
    bars = ax.bar([t.capitalize() for t in terrains], final_avgs,
                  color=[colors[t] for t in terrains], alpha=0.85)
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_ylabel("Avg Final Reward (last 20 eps)")
    ax.set_title("Final Performance by Terrain")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("ppo_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved → ppo_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN  (if __name__ guard is required on Windows for DummyVecEnv)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"PyTorch  : {torch.__version__}")
    print(f"Device   : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU      : {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING  : CUDA not found — running on CPU.")
        print("           Install CUDA PyTorch:")
        print("           pip install torch --index-url https://download.pytorch.org/whl/cu128")

    terrains     = ["flat"]  # , "rough", "slope"]
    all_rewards  = {}

    # ── Training ──
    for terrain in terrains:
        rewards              = train_terrain(terrain, total_timesteps=3_000_000)
        all_rewards[terrain] = rewards

    with open("ppo_rewards.pkl", "wb") as f:
        pickle.dump(all_rewards, f)
    print("\nRewards saved → ppo_rewards.pkl")

    # ── Evaluation ──
    print("\n=== EVALUATION ===")
    for terrain in terrains:
        evaluate(terrain, n_episodes=5)

    # ── Summary table ──
    print("\n=== SUMMARY TABLE ===")
    print(f"{'Terrain':<10} {'Final Avg Reward':<20} {'Episodes trained'}")
    print("-" * 50)
    for terrain in terrains:
        r = all_rewards[terrain]
        print(f"{terrain:<10} {np.mean(r[-20:]):<20.2f} {len(r)}")

    # ── Plots ──
    plot_results(all_rewards)
