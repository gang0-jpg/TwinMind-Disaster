from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class FloodNPZDataset(Dataset):
    def __init__(
        self,
        cases_dir: str | Path,
        rain_clip: float = 100.0,
        use_mask: bool = True,
    ) -> None:
        self.cases_dir = Path(cases_dir)
        self.rain_clip = rain_clip
        self.use_mask = use_mask

        self.files = sorted(self.cases_dir.glob("case_*.npz"))
        if not self.files:
            raise RuntimeError(f"No NPZ files found in {self.cases_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def _normalize_dem(self, dem: np.ndarray, mask: np.ndarray) -> np.ndarray:
        valid = mask == 1
        if valid.sum() == 0:
            mean = float(dem.mean())
            std = float(dem.std()) + 1e-6
        else:
            mean = float(dem[valid].mean())
            std = float(dem[valid].std()) + 1e-6
        dem_norm = (dem - mean) / std
        return dem_norm.astype(np.float32)

    def _normalize_rain(self, rain: np.ndarray) -> np.ndarray:
        rain = np.clip(rain, 0.0, self.rain_clip) / self.rain_clip
        return rain.astype(np.float32)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        d = np.load(path, allow_pickle=True)

        dem = d["dem"].astype(np.float32)
        rain = d["rain"].astype(np.float32)
        flood_gt = d["flood_gt"].astype(np.float32)
        mask = d["mask"].astype(np.uint8)

        dem = self._normalize_dem(dem, mask)
        rain = self._normalize_rain(rain)

        x = np.concatenate(
            [
                dem[None, :, :],
                rain,
            ],
            axis=0,
        ).astype(np.float32)

        y = flood_gt[None, :, :].astype(np.float32)

        sample = {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "mask": torch.from_numpy(mask[None, :, :].astype(np.float32)),
            "path": str(path),
        }

        if self.use_mask:
            sample["y"] = sample["y"] * sample["mask"]

        return sample
