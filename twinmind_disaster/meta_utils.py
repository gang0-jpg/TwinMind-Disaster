from datetime import datetime, UTC


def build_case_meta(
    case_id: int,
    tile_id: int,
    rain_meta: dict,
    solver_meta: dict,
    config,
    seed: int,
) -> dict:
    meta = {
        "case_id": f"{case_id:06d}",
        "spec_version": config.spec_version,
        "dataset_version": config.dataset_version,
        "task": config.task,
        "region_name": f"{config.region_name_prefix}_{tile_id:04d}",
        "grid_height": config.grid_height,
        "grid_width": config.grid_width,
        "grid_resolution_m": config.grid_resolution_m,
        "t_steps": config.t_steps,
        "dt_min": config.dt_min,
        "rain_unit": "mm_per_hour",
        "flood_unit": "m",
        "dem_source": "gsi_dem",
        "seed": seed,
        "created_at": datetime.now(UTC).isoformat(),
    }
    meta.update(rain_meta)
    meta.update(solver_meta)
    return meta
