from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # =========================
    # Paths
    # =========================
    project_root: Path = Path(".")
    data_dir: Path = Path("data")
    raw_dem_dir: Path = Path("data/raw_dem")
    raw_mask_dir: Path = Path("data/raw_mask")
    cases_dir: Path = Path("data/cases")
    preview_dir: Path = Path("data/preview")
    logs_dir: Path = Path("logs")

    # =========================
    # Dataset / spec
    # =========================
    spec_version: str = "twm_obs_case_v1"
    dataset_version: str = "twm_disaster_flood_v1.0"
    task: str = "flood"
    region_name_prefix: str = "virtual_tile"

    # =========================
    # Grid
    # =========================
    grid_height: int = 256
    grid_width: int = 256
    grid_resolution_m: float = 10.0

    # =========================
    # Time
    # =========================
    t_steps: int = 6
    dt_min: int = 10

    # =========================
    # Rain generator
    # =========================
    rain_generator_name: str = "synthetic_rain_v1"

    scenario_probs: dict = None

    rain_total_min_mm: float = 80.0
    rain_total_max_mm: float = 250.0

    rain_peak_min_mmph: float = 20.0
    rain_peak_max_mmph: float = 100.0

    sigma_min_px: float = 15.0
    sigma_max_px: float = 45.0

    rain_clip_max_mmph: float = 120.0
    noise_ratio: float = 0.05

    # =========================
    # Flood solver
    # =========================
    flood_solver_name: str = "simple_cell_diffusion_v1"
    flow_alpha: float = 0.20
    inner_steps: int = 4
    neighbor_mode: int = 4
    evap: float = 0.0
    flood_clip_max_m: float = 5.0

    # =========================
    # Preview
    # =========================
    save_preview_every: int = 50
    save_preview_first_n: int = 20

    def __post_init__(self) -> None:
        if self.scenario_probs is None:
            self.scenario_probs = {
                "moving_gaussian": 0.40,
                "stationary_gaussian": 0.25,
                "uniform_heavy": 0.15,
                "dual_peak": 0.20,
            }

    def ensure_dirs(self) -> None:
        for p in [
            self.data_dir,
            self.raw_dem_dir,
            self.raw_mask_dir,
            self.cases_dir,
            self.preview_dir,
            self.logs_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)
