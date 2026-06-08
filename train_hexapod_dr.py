"""
Hexapod PPO training with Domain Randomization.

Each episode randomly picks a terrain type (flat/rough/slope/stairs/hills)
and a difficulty level (0.1–0.8), so the policy learns to walk on all terrains.

Saved model: ppo_hexapod_dr.zip

Run:
    venv\Scripts\python train_hexapod_dr.py

View result:
    venv\Scripts\python view_ppo_terrain.py flat     --model ppo_hexapod_dr
    venv\Scripts\python view_ppo_terrain.py stairs   --model ppo_hexapod_dr
"""

import os
import random
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

from terrain_generator import TerrainGenerator

# ─────────────────────────────────────────────────────────────────────────────
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

TERRAIN_TYPES = ["flat", "rough", "slope", "stairs", "hills"]
DEVICE        = "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
class HexapodDREnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.m          = mujoco.MjModel.from_xml_path(XML_PATH)
        self.d          = mujoco.MjData(self.m)
        self.tg         = TerrainGenerator(self.m)
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

        # ── randomise terrain each episode ──────────────────────────────────
        terrain    = random.choice(TERRAIN_TYPES)
        difficulty = random.uniform(0.1, 0.8)

        if terrain == "flat":
            self.tg.flat()
        elif terrain == "rough":
            self.tg.rough(difficulty=difficulty)
        elif terrain == "slope":
            self.tg.slope(difficulty=difficulty)
        elif terrain == "stairs":
            self.tg.stairs(difficulty=difficulty)
        elif terrain == "hills":
            self.tg.hills(difficulty=difficulty)

        # spawn robot on top of terrain at the origin
        mujoco.mj_resetData(self.m, self.d)
        spawn_z         = self.tg.height_at(0.0, 0.0) + 0.28
        self.d.qpos[2]  = spawn_z
        self.d.qpos[7:] = NOMINAL.copy()
        mujoco.mj_forward(self.m, self.d)
        self._settle()

        self.step_count = 0
        self.prev_x     = self.d.qpos[0]
        return self._get_obs(), {}

    def step(self, action):
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

        # tighter fall threshold to account for terrain height variation
        terrain_h  = self.tg.height_at(self.d.qpos[0], self.d.qpos[1])
        body_above = height - terrain_h
        terminated = bool(body_above < 0.08 or tilt > 1.0)
        truncated  = bool(self.step_count >= self.max_steps)
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK
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
# CHECKPOINT RESUME
# ─────────────────────────────────────────────────────────────────────────────
def _latest_checkpoint(ckpt_dir, prefix):
    if not os.path.isdir(ckpt_dir):
        return None
    files = [f for f in os.listdir(ckpt_dir)
             if f.startswith(prefix) and f.endswith(".zip")]
    if not files:
        return None

    def step_count(name):
        try:
            return int(name.replace(prefix + "_", "").replace("_steps.zip", ""))
        except ValueError:
            return -1

    best = max(files, key=step_count)
    return os.path.join(ckpt_dir, best), step_count(best)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────
def train(total_timesteps=5_000_000):
    print(f"\n{'='*55}")
    print(f"  Domain Randomization PPO  |  device: {DEVICE}")
    print(f"  Terrains: {TERRAIN_TYPES}")
    print(f"  Difficulty: 0.1 – 0.8 (random each episode)")
    print(f"{'='*55}")

    ckpt_dir    = os.path.join("checkpoints", "dr")
    ckpt_prefix = "ppo_hexapod_dr"
    os.makedirs(ckpt_dir, exist_ok=True)

    env           = DummyVecEnv([HexapodDREnv])
    reward_logger = RewardLogger()
    checkpoint_cb = CheckpointCallback(
        save_freq   = 50_000,
        save_path   = ckpt_dir,
        name_prefix = ckpt_prefix,
        verbose     = 1,
    )
    callbacks = CallbackList([reward_logger, checkpoint_cb])

    resume = _latest_checkpoint(ckpt_dir, ckpt_prefix)
    if resume:
        ckpt_path, steps_done = resume
        remaining = total_timesteps - steps_done
        if remaining <= 0:
            print(f"  Already trained {steps_done:,} steps — skipping.")
            env.close()
            return []
        print(f"  Resuming from: {ckpt_path}  ({steps_done:,} / {total_timesteps:,} steps done)")
        model = PPO.load(ckpt_path, env=env, device=DEVICE,
                         tensorboard_log="./logs/dr")
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
            tensorboard_log = "./logs/dr",
        )

    model.learn(
        total_timesteps     = remaining,
        callback            = callbacks,
        progress_bar        = True,
        reset_num_timesteps = resume is None,
    )

    model.save("ppo_hexapod_dr")
    print(f"\n  Model saved → ppo_hexapod_dr.zip")
    if reward_logger.episode_rewards:
        print(f"  Final avg reward (last 20 eps): "
              f"{np.mean(reward_logger.episode_rewards[-20:]):.2f}")

    env.close()
    return reward_logger.episode_rewards


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(n_episodes=10):
    model  = PPO.load("ppo_hexapod_dr", device=DEVICE)
    env    = HexapodDREnv()
    counts = {t: {"rewards": [], "distances": []} for t in TERRAIN_TYPES}

    for ep in range(n_episodes):
        obs, _     = env.reset()
        # find out which terrain was chosen (approximate — first step)
        done       = False
        total_r    = 0.0
        start_x    = env.d.qpos[0]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total_r  += r
            done      = terminated or truncated

        dist = env.d.qpos[0] - start_x
        print(f"  ep {ep+1:2d}: reward={total_r:7.1f}  dist={dist:.3f} m")

    env.close()
    print(f"\n  Avg reward : {np.mean([r for t in counts.values() for r in t['rewards']]):.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(rewards):
    if not rewards:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    smoothed = np.convolve(rewards, np.ones(30)/30, mode="valid")
    ax.plot(rewards, alpha=0.2, color="#2196F3", linewidth=0.8, label="Raw")
    ax.plot(smoothed, color="#2196F3", linewidth=2, label="30-ep rolling avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Domain Randomization Training — All Terrains")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("dr_training_curve.png", dpi=150, bbox_inches="tight")
    print("Plot saved → dr_training_curve.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"PyTorch : {torch.__version__}  |  Device: {DEVICE}")

    rewards = train(total_timesteps=5_000_000)

    with open("dr_rewards.pkl", "wb") as f:
        pickle.dump(rewards, f)
    print("Rewards saved → dr_rewards.pkl")

    print("\n=== EVALUATION ===")
    evaluate(n_episodes=10)

    plot_results(rewards)
