import torch
import torch.nn as nn

from twinmind_disaster.model_unet import UNetSmall
from twinmind_disaster.model_reservoir import RainReservoirEncoder


class ReservoirUNet(nn.Module):
    def __init__(
        self,
        reservoir_dim: int = 16,
        base_channels: int = 32,
        rain_timesteps: int = 5,
    ) -> None:
        super().__init__()
        self.reservoir_dim = reservoir_dim
        self.rain_timesteps = rain_timesteps

        self.rain_encoder = RainReservoirEncoder(
            input_dim=1,
            hidden_dim=reservoir_dim,
            spectral_scale=0.9,
            input_scale=0.5,
            leak_rate=1.0,
        )

        # DEM(1) + slope(1) + rainfall(6) + reservoir_dim
        in_channels = 2 + rain_timesteps + reservoir_dim

        self.unet = UNetSmall(
            in_channels=in_channels,
            out_channels=1,
            base_ch=base_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 8, H, W)
           channel 0: DEM
           channel 1: slope
           channel 2..7: rainfall 6 timesteps
        """
        dem = x[:, 0:1, :, :]
        slope = x[:, 1:2, :, :]
        rain_maps = x[:, 2:, :, :]   # (B, 6, H, W)

        # 各時刻の空間平均雨量をReservoirへ
        rain_seq = rain_maps.mean(dim=(2, 3))  # (B, 6)
        state = self.rain_encoder(rain_seq)    # (B, reservoir_dim)

        bsz, _, h, w = x.shape
        state_maps = state.unsqueeze(-1).unsqueeze(-1).expand(
            bsz, self.reservoir_dim, h, w
        )

        # 元の rainfall(6ch) は消さずに残す
        fused = torch.cat([dem, slope, rain_maps, state_maps], dim=1)
        return self.unet(fused)
