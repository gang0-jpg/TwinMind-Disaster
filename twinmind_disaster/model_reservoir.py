import torch
import torch.nn as nn


class RainReservoirEncoder(nn.Module):
    """
    Rainfall time series (B, T) or (B, T, 1) を受け取り、
    reservoir state (B, hidden_dim) を返す。
    reservoir本体は固定、readoutは使わず state をそのまま出力。
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 16,
        spectral_scale: float = 0.9,
        input_scale: float = 0.5,
        leak_rate: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.leak_rate = leak_rate

        w_in = torch.randn(hidden_dim, input_dim) * input_scale
        w = torch.randn(hidden_dim, hidden_dim)

        # スペクトル半径をおおまかに調整
        eigvals = torch.linalg.eigvals(w).abs()
        max_eig = torch.max(eigvals).real.clamp(min=1e-6)
        w = w / max_eig * spectral_scale

        self.register_buffer("w_in", w_in)
        self.register_buffer("w", w)

    def forward(self, rain_seq: torch.Tensor) -> torch.Tensor:
        """
        rain_seq:
            (B, T) または (B, T, 1)
        returns:
            state: (B, hidden_dim)
        """
        if rain_seq.dim() == 2:
            rain_seq = rain_seq.unsqueeze(-1)

        bsz, timesteps, _ = rain_seq.shape
        device = rain_seq.device
        x = torch.zeros(bsz, self.hidden_dim, device=device)

        for t in range(timesteps):
            u_t = rain_seq[:, t, :]  # (B, input_dim)
            pre_act = u_t @ self.w_in.T + x @ self.w.T
            x_new = torch.tanh(pre_act)
            x = (1.0 - self.leak_rate) * x + self.leak_rate * x_new

        return x
