import numpy as np

from twinmind_disaster.config import Config


def _zero_boundary_flows(
    flow_up: np.ndarray,
    flow_down: np.ndarray,
    flow_left: np.ndarray,
    flow_right: np.ndarray,
) -> None:
    flow_up[0, :] = 0.0
    flow_down[-1, :] = 0.0
    flow_left[:, 0] = 0.0
    flow_right[:, -1] = 0.0


def _neighbor_masks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask_up = np.roll(mask, shift=1, axis=0)
    mask_down = np.roll(mask, shift=-1, axis=0)
    mask_left = np.roll(mask, shift=1, axis=1)
    mask_right = np.roll(mask, shift=-1, axis=1)

    # boundary invalidation
    mask_up[0, :] = 0
    mask_down[-1, :] = 0
    mask_left[:, 0] = 0
    mask_right[:, -1] = 0

    return mask_up, mask_down, mask_left, mask_right


def simulate_flood(
    dem: np.ndarray,
    rain: np.ndarray,
    mask: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, dict]:
    if config.neighbor_mode != 4:
        raise NotImplementedError("Only 4-neighbor mode is implemented in v1.")

    water = np.zeros_like(dem, dtype=np.float32)
    valid = (mask == 1).astype(np.uint8)

    mask_up, mask_down, mask_left, mask_right = _neighbor_masks(mask)

    for t in range(config.t_steps):
        rain_add = rain[t].astype(np.float32) / 1000.0 * (config.dt_min / 60.0)
        water[valid == 1] += rain_add[valid == 1]
        water[valid == 0] = 0.0

        for _ in range(config.inner_steps):
            head = dem + water

            head_up = np.roll(head, shift=1, axis=0)
            head_down = np.roll(head, shift=-1, axis=0)
            head_left = np.roll(head, shift=1, axis=1)
            head_right = np.roll(head, shift=-1, axis=1)

            flow_up = config.flow_alpha * np.maximum(head - head_up, 0.0)
            flow_down = config.flow_alpha * np.maximum(head - head_down, 0.0)
            flow_left = config.flow_alpha * np.maximum(head - head_left, 0.0)
            flow_right = config.flow_alpha * np.maximum(head - head_right, 0.0)

            # boundary wrap-around cancel
            _zero_boundary_flows(flow_up, flow_down, flow_left, flow_right)

            # invalid source cell
            flow_up[valid == 0] = 0.0
            flow_down[valid == 0] = 0.0
            flow_left[valid == 0] = 0.0
            flow_right[valid == 0] = 0.0

            # invalid destination cell
            flow_up[mask_up == 0] = 0.0
            flow_down[mask_down == 0] = 0.0
            flow_left[mask_left == 0] = 0.0
            flow_right[mask_right == 0] = 0.0

            outflow = flow_up + flow_down + flow_left + flow_right

            # limit outflow by available water
            scale = np.ones_like(outflow, dtype=np.float32)
            positive = outflow > 1e-8
            scale[positive] = np.minimum(1.0, water[positive] / outflow[positive])

            flow_up *= scale
            flow_down *= scale
            flow_left *= scale
            flow_right *= scale

            outflow = flow_up + flow_down + flow_left + flow_right

            inflow = (
                np.roll(flow_down, shift=1, axis=0) +
                np.roll(flow_up, shift=-1, axis=0) +
                np.roll(flow_right, shift=1, axis=1) +
                np.roll(flow_left, shift=-1, axis=1)
            )

            # cancel wrapped inflow
            inflow[0, :] -= flow_down[-1, :]
            inflow[-1, :] -= flow_up[0, :]
            inflow[:, 0] -= flow_right[:, -1]
            inflow[:, -1] -= flow_left[:, 0]

            water = water - outflow + inflow

            if config.evap > 0.0:
                water *= (1.0 - config.evap)

            water = np.maximum(water, 0.0)
            water[valid == 0] = 0.0

    flood_gt = np.clip(water, 0.0, config.flood_clip_max_m).astype(np.float16)

    solver_meta = {
        "flood_solver": config.flood_solver_name,
        "flow_alpha": config.flow_alpha,
        "inner_steps": config.inner_steps,
        "neighbor_mode": config.neighbor_mode,
        "evap": config.evap,
    }

    return flood_gt, solver_meta
