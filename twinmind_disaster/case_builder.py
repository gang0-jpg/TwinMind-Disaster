import numpy as np
from pathlib import Path

from twinmind_disaster.dem_utils import load_dem_tile, load_mask_tile, validate_dem_and_mask
from twinmind_disaster.flood_solver import simulate_flood
from twinmind_disaster.io_utils import save_case_npz, save_case_preview
from twinmind_disaster.meta_utils import build_case_meta
from twinmind_disaster.rain_generator import sample_scenario_type, generate_rain_case


def build_single_case(
    case_id: int,
    tile_id: int,
    config,
    save_preview: bool = False,
    seed: int | None = None,
) -> dict:
    if seed is None:
        seed = case_id

    rng = np.random.default_rng(seed)

    dem = load_dem_tile(tile_id, config)
    mask = load_mask_tile(tile_id, config)
    validate_dem_and_mask(dem, mask, config)

    scenario_type = sample_scenario_type(rng, config)
    rain, rain_meta = generate_rain_case(scenario_type, rng, config)

    flood_gt, solver_meta = simulate_flood(dem, rain, mask, config)

    meta = build_case_meta(
        case_id=case_id,
        tile_id=tile_id,
        rain_meta=rain_meta,
        solver_meta=solver_meta,
        config=config,
        seed=seed,
    )

    output_path = config.cases_dir / f"case_{case_id:06d}.npz"
    save_case_npz(output_path, dem, rain, flood_gt, mask, meta)

    if save_preview:
        save_case_preview(config.preview_dir, case_id, dem, rain, flood_gt)

    return {
        "case_id": f"{case_id:06d}",
        "tile_id": f"{tile_id:04d}",
        "scenario_type": scenario_type,
        "seed": seed,
        "output_path": str(output_path),
    }
