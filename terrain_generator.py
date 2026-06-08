"""
Terrain generator for the hexapod heightfield.

The XML declares:  <hfield name="terrain" nrow="128" ncol="128" size="5 5 0.5 0.1"/>
  - grid  : 128 × 128 cells → 10 m × 10 m world footprint
  - z_scale: 0.5 m  (data value 1.0 == 0.5 m above floor)
  - base   : 0.1 m  (solid below the lowest point)

Usage:
    import mujoco
    from terrain_generator import TerrainGenerator

    m = mujoco.MjModel.from_xml_path("...hexapod_terrain.xml")
    tg = TerrainGenerator(m)

    tg.rough(difficulty=0.5)   # writes into m.hfield_data
    spawn_z = tg.height_at(0, 0) + 0.20   # stand 20 cm above ground

Available terrain types
-----------------------
flat      – perfectly flat floor (default / easiest)
rough     – smoothed random noise  (difficulty 0–1 → bump height 0–8 cm)
slope     – constant incline along x (difficulty 0–1 → 0–15° angle)
stairs    – forward-facing steps    (difficulty 0–1 → step height 0–6 cm)
hills     – Gaussian mounds scattered across the field
random    – picks one of the above at random
"""

import numpy as np
from scipy.ndimage import gaussian_filter


class TerrainGenerator:
    def __init__(self, model):
        self.model   = model
        hid          = mujoco_hfield_id(model, "terrain")
        self.nrow    = model.hfield_nrow[hid]
        self.ncol    = model.hfield_ncol[hid]
        self.z_scale = model.hfield_size[hid, 2]   # 0.5 m
        self._adr    = model.hfield_adr[hid]        # offset into hfield_data

        # physical half-extents (metres)
        self.x_half  = model.hfield_size[hid, 0]   # 5 m
        self.y_half  = model.hfield_size[hid, 1]   # 5 m

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC TERRAIN METHODS
    # ─────────────────────────────────────────────────────────────────────

    def flat(self):
        self._write(np.zeros((self.nrow, self.ncol)))

    def rough(self, difficulty: float = 0.5, seed: int = None):
        """
        Smoothed random noise.
        difficulty 0 → amplitude 0 cm (flat)
        difficulty 1 → amplitude 8 cm
        """
        rng       = np.random.default_rng(seed)
        noise     = rng.random((self.nrow, self.ncol))
        smoothed  = gaussian_filter(noise, sigma=4.0)
        smoothed -= smoothed.min()
        smoothed /= smoothed.max() + 1e-8
        max_h     = 0.08 * difficulty          # metres
        self._write(smoothed * (max_h / self.z_scale))

    def slope(self, difficulty: float = 0.5):
        """
        Constant upward incline in the +x direction.
        difficulty 0 → 0°,  difficulty 1 → 15°
        """
        angle_deg = 15.0 * difficulty
        angle_rad = np.deg2rad(angle_deg)
        # height rises with column index (x increases left→right)
        x         = np.linspace(-self.x_half, self.x_half, self.ncol)
        height_m  = np.tan(angle_rad) * (x - x.min())
        grid      = np.tile(height_m, (self.nrow, 1))
        grid     /= self.z_scale
        self._write(np.clip(grid, 0, 1))

    def stairs(self, difficulty: float = 0.5, n_steps: int = 8):
        """
        Discrete steps that rise in the +x direction.
        difficulty 0 → step height 0 cm,  difficulty 1 → step height 6 cm
        """
        step_h_m  = 0.06 * difficulty          # metres per step
        cols      = np.arange(self.ncol)
        step_idx  = (cols / self.ncol * n_steps).astype(int)
        height_m  = step_idx * step_h_m
        grid      = np.tile(height_m, (self.nrow, 1))
        grid     /= self.z_scale
        self._write(np.clip(grid, 0, 1))

    def hills(self, difficulty: float = 0.5, n_hills: int = 12, seed: int = None):
        """
        Gaussian mounds scattered randomly across the field.
        difficulty 0 → height 0 cm,  difficulty 1 → height 12 cm
        """
        rng      = np.random.default_rng(seed)
        grid     = np.zeros((self.nrow, self.ncol))
        max_h    = 0.12 * difficulty
        rows     = np.linspace(0, 1, self.nrow)
        cols     = np.linspace(0, 1, self.ncol)
        C, R     = np.meshgrid(cols, rows)

        for _ in range(n_hills):
            cx  = rng.random()
            cy  = rng.random()
            sig = rng.uniform(0.05, 0.15)
            amp = rng.uniform(0.5, 1.0) * max_h / self.z_scale
            grid += amp * np.exp(-((C - cx)**2 + (R - cy)**2) / (2 * sig**2))

        grid -= grid.min()
        self._write(np.clip(grid, 0, 1))

    def random(self, difficulty: float = 0.5, seed: int = None):
        rng     = np.random.default_rng(seed)
        terrain = rng.choice(["rough", "slope", "stairs", "hills"])
        child_seed = int(rng.integers(0, 2**31))
        print(f"  [TerrainGenerator] random → '{terrain}'  difficulty={difficulty:.2f}")
        getattr(self, terrain)(difficulty=difficulty, seed=child_seed
                               if terrain in ("rough", "hills") else None)
        return terrain

    # ─────────────────────────────────────────────────────────────────────
    # UTILITY
    # ─────────────────────────────────────────────────────────────────────

    def height_at(self, world_x: float, world_y: float) -> float:
        """
        Return terrain surface height (metres, world frame) at (world_x, world_y).
        Useful for computing the hexapod's spawn z position.
        """
        # column: x maps left→right,  row: y maps top→bottom in data array
        col = int((world_x + self.x_half) / (2 * self.x_half) * (self.ncol - 1))
        row = int((self.y_half - world_y) / (2 * self.y_half) * (self.nrow - 1))
        col = np.clip(col, 0, self.ncol - 1)
        row = np.clip(row, 0, self.nrow - 1)
        data = self.model.hfield_data[self._adr : self._adr + self.nrow * self.ncol]
        return float(data[row * self.ncol + col]) * self.z_scale

    def _write(self, grid: np.ndarray):
        """Write normalised [0, 1] grid into model.hfield_data."""
        flat = np.clip(grid, 0.0, 1.0).ravel().astype(np.float32)
        n    = self.nrow * self.ncol
        self.model.hfield_data[self._adr : self._adr + n] = flat


def mujoco_hfield_id(model, name: str) -> int:
    """Return the index of an hfield by name."""
    import mujoco
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, name)
    if hid < 0:
        raise ValueError(f"hfield '{name}' not found in model")
    return hid
