# Hexapod Locomotion: CPG vs PPO

Comparing Central Pattern Generator (CPG) and Reinforcement Learning (PPO) controllers for hexapod locomotion in MuJoCo.

## Setup

```bash
# 1. Clone this repo
git clone <your-repo-url>
cd <repo-folder>

# 2. Clone the hexapod model (required)
git clone https://github.com/TemplierPaul/one_policy_to_run_them_all

# 3. Create virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate
pip install mujoco stable-baselines3 scipy matplotlib tensorboard tqdm rich
```

## Files

| File | Description |
|------|-------------|
| `view_hexapod_cpg.py` | CPG tripod gait viewer — runs robot for 10 m with deviation tracking |
| `view_ppo.py` | View a trained PPO policy on flat terrain |
| `view_ppo_terrain.py` | View PPO policy on any terrain (`flat`, `rough`, `slope`, `stairs`, `hills`) |
| `view_hexapod_terrain.py` | Interactive terrain viewer (keyboard switching) |
| `view_parallel.py` | Side-by-side comparison of CPG, PPO-flat, and PPO-DR |
| `train_hexapod_ppo.py` | Train PPO on flat terrain |
| `train_hexapod_dr.py` | Train PPO with domain randomization (all terrains) |
| `compare_cpg_ppo.py` | Headless evaluation — prints comparison table and saves plots |
| `terrain_generator.py` | MuJoCo heightfield terrain generator |

## Usage

### View CPG walking
```bash
venv\Scripts\python view_hexapod_cpg.py
```

### Train PPO (flat terrain)
```bash
venv\Scripts\python train_hexapod_ppo.py
```

### Train PPO with domain randomization
```bash
venv\Scripts\python train_hexapod_dr.py
```

### View trained PPO
```bash
venv\Scripts\python view_ppo.py
venv\Scripts\python view_ppo_terrain.py stairs
venv\Scripts\python view_ppo_terrain.py slope --model ppo_hexapod_dr
```

### Run full comparison
```bash
venv\Scripts\python compare_cpg_ppo.py
```

### Side-by-side parallel viewer
```bash
venv\Scripts\python view_parallel.py
```

## Results (Flat Terrain — 10 m run)

| Controller | Speed | Max Y deviation | Notes |
|------------|-------|-----------------|-------|
| CPG Tripod | 0.15 m/s | 0.26 m | Open-loop, no training |
| PPO Flat | 1.08 m/s | 0.56 m | 3M steps, flat only |
| PPO DR | TBD | TBD | 5M steps, all terrains |
