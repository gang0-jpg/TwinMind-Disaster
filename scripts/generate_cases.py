import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import traceback
from pathlib import Path

import numpy as np

from twinmind_disaster.config import Config
from twinmind_disaster.case_builder import build_single_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cases", type=int, required=True)
    parser.add_argument("--start_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


def discover_tile_ids(raw_dem_dir: Path) -> list[int]:
    tile_ids = []
    for p in raw_dem_dir.glob("dem_tile_*.npy"):
        stem = p.stem  # dem_tile_0001
        try:
            tile_ids.append(int(stem.split("_")[-1]))
        except ValueError:
            continue
    tile_ids.sort()
    return tile_ids


def should_save_preview(case_id: int, config: Config, preview_flag: bool) -> bool:
    if not preview_flag:
        return False
    if case_id <= config.save_preview_first_n:
        return True
    return case_id % config.save_preview_every == 0


def main() -> None:
    args = parse_args()

    config = Config()
    config.ensure_dirs()

    tile_ids = discover_tile_ids(config.raw_dem_dir)
    if not tile_ids:
        raise RuntimeError(f"No DEM tiles found in {config.raw_dem_dir}")

    rng = np.random.default_rng(args.seed)

    log_path = config.logs_dir / "generation.log"
    with open(log_path, "a", encoding="utf-8") as logf:
        for case_id in range(args.start_id, args.start_id + args.num_cases):
            tile_id = int(rng.choice(tile_ids))
            seed = int(rng.integers(0, 2**31 - 1))

            try:
                result = build_single_case(
                    case_id=case_id,
                    tile_id=tile_id,
                    config=config,
                    save_preview=should_save_preview(case_id, config, args.preview),
                    seed=seed,
                )
                msg = (
                    f"[OK] case_{result['case_id']} "
                    f"tile={result['tile_id']} "
                    f"scenario={result['scenario_type']} "
                    f"seed={result['seed']}"
                )
                print(msg)
                logf.write(msg + "\n")
                logf.flush()

            except Exception as e:
                msg = f"[ERR] case_{case_id:06d} tile={tile_id:04d} error={e}"
                print(msg)
                logf.write(msg + "\n")
                logf.write(traceback.format_exc() + "\n")
                logf.flush()
                continue


if __name__ == "__main__":
    main()
