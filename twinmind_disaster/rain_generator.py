import numpy as np

from twinmind_disaster.config import Config


def sample_scenario_type(rng: np.random.Generator, config: Config) -> str:
    names = list(config.scenario_probs.keys())
    probs = np.array(list(config.scenario_probs.values()), dtype=np.float64)
    probs = probs / probs.sum()
    return rng.choice(names, p=probs)


def _time_profile(kind: str) -> np.ndarray:
    if kind == "moving_gaussian":
        return np.array([0.4, 0.7, 1.0, 1.0, 0.7, 0.4], dtype=np.float32)
    if kind == "stationary_gaussian":
        return np.array([0.5, 0.8, 1.0, 1.0, 0.8, 0.5], dtype=np.float32)
    if kind == "uniform_heavy":
        return np.array([0.6, 0.8, 1.0, 1.0, 0.8, 0.6], dtype=np.float32)
    if kind == "dual_peak":
        return np.array([0.5, 0.8, 1.0, 1.0, 0.8, 0.5], dtype=np.float32)
    raise ValueError(f"Unknown scenario_type: {kind}")


def _meshgrid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.arange(h, dtype=np.float32)
    x = np.arange(w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def _gaussian_2d(
    xx: np.ndarray,
    yy: np.ndarray,
    cx: float,
    cy: float,
    sigma: float,
    amplitude: float,
) -> np.ndarray:
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return amplitude * np.exp(-dist2 / (2.0 * sigma * sigma))


def _apply_noise(field: np.ndarray, rng: np.random.Generator, noise_ratio: float) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=noise_ratio, size=field.shape).astype(np.float32)
    out = field * (1.0 + noise)
    return np.clip(out, 0.0, None)


def _rescale_rain_total(rain: np.ndarray, target_total_mm: float, dt_min: int) -> np.ndarray:
    # rain unit = mm/h
    # total per pixel across time in mm:
    # sum_t rain[t] * dt/60
    current_total = float(np.mean(np.sum(rain, axis=0) * (dt_min / 60.0)))
    if current_total <= 1e-8:
        return rain
    scale = target_total_mm / current_total
    return rain * scale


def generate_rain_case(
    scenario_type: str,
    rng: np.random.Generator,
    config: Config,
) -> tuple[np.ndarray, dict]:
    h, w = config.grid_height, config.grid_width
    t_steps = config.t_steps
    xx, yy = _meshgrid(h, w)

    rain_total_mm = float(rng.uniform(config.rain_total_min_mm, config.rain_total_max_mm))
    rain_peak_mmph = float(rng.uniform(config.rain_peak_min_mmph, config.rain_peak_max_mmph))
    profile = _time_profile(scenario_type)

    rain = np.zeros((t_steps, h, w), dtype=np.float32)

    rain_meta: dict = {
        "rain_generator": config.rain_generator_name,
        "scenario_type": scenario_type,
        "rain_total_mm": rain_total_mm,
        "rain_peak_mmph": rain_peak_mmph,
    }

    if scenario_type == "moving_gaussian":
        sigma = float(rng.uniform(config.sigma_min_px, config.sigma_max_px))
        x0, y0 = float(rng.uniform(0, w - 1)), float(rng.uniform(0, h - 1))
        x1, y1 = float(rng.uniform(0, w - 1)), float(rng.uniform(0, h - 1))

        for t in range(t_steps):
            r = t / max(t_steps - 1, 1)
            cx = (1 - r) * x0 + r * x1
            cy = (1 - r) * y0 + r * y1
            amp = rain_peak_mmph * profile[t]
            rain[t] = _gaussian_2d(xx, yy, cx, cy, sigma, amp)
            rain[t] = _apply_noise(rain[t], rng, config.noise_ratio)

        rain_meta.update({
            "storm_sigma_px": sigma,
            "storm_start_xy": [x0, y0],
            "storm_end_xy": [x1, y1],
        })

    elif scenario_type == "stationary_gaussian":
        sigma = float(rng.uniform(config.sigma_min_px, config.sigma_max_px))
        cx, cy = float(rng.uniform(0, w - 1)), float(rng.uniform(0, h - 1))

        for t in range(t_steps):
            amp = rain_peak_mmph * profile[t]
            rain[t] = _gaussian_2d(xx, yy, cx, cy, sigma, amp)
            rain[t] = _apply_noise(rain[t], rng, config.noise_ratio)

        rain_meta.update({
            "storm_sigma_px": sigma,
            "storm_center_xy": [cx, cy],
        })

    elif scenario_type == "uniform_heavy":
        for t in range(t_steps):
            amp = rain_peak_mmph * profile[t]
            base = np.full((h, w), amp, dtype=np.float32)
            rain[t] = _apply_noise(base, rng, config.noise_ratio)

    elif scenario_type == "dual_peak":
        sigma1 = float(rng.uniform(config.sigma_min_px, config.sigma_max_px))
        sigma2 = float(rng.uniform(config.sigma_min_px, config.sigma_max_px))

        x1, y1 = float(rng.uniform(0, w - 1)), float(rng.uniform(0, h - 1))
        x2, y2 = float(rng.uniform(0, w - 1)), float(rng.uniform(0, h - 1))

        for t in range(t_steps):
            amp1 = rain_peak_mmph * profile[t] * float(rng.uniform(0.4, 0.7))
            amp2 = rain_peak_mmph * profile[t] * float(rng.uniform(0.4, 0.7))
            r1 = _gaussian_2d(xx, yy, x1, y1, sigma1, amp1)
            r2 = _gaussian_2d(xx, yy, x2, y2, sigma2, amp2)
            rain[t] = _apply_noise(r1 + r2, rng, config.noise_ratio)

        rain_meta.update({
            "storm1_sigma_px": sigma1,
            "storm2_sigma_px": sigma2,
            "storm1_center_xy": [x1, y1],
            "storm2_center_xy": [x2, y2],
        })

    else:
        raise ValueError(f"Unsupported scenario_type: {scenario_type}")

    rain = np.clip(rain, 0.0, config.rain_clip_max_mmph)
    rain = _rescale_rain_total(rain, rain_total_mm, config.dt_min)
    rain = np.clip(rain, 0.0, config.rain_clip_max_mmph)

    return rain.astype(np.float16), rain_meta
