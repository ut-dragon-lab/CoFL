"""CoFLSFieldDecoder query decoder, independent of training and simulation frameworks."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from cofl.runtime.inference_rules import MODEL_HFOV_MAX_DEG, MODEL_HFOV_MIN_DEG

from .blocks import GaussianFourierEncoding, PreNormDecoderBlock

THETA_REF_RAD = math.pi / 2
MODEL_HFOV_MIN_RAD = math.radians(MODEL_HFOV_MIN_DEG)
MODEL_HFOV_MAX_RAD = math.radians(MODEL_HFOV_MAX_DEG)


def clamp_hfov_for_model(hfov_rad: float) -> float:
    """Clamp model conditioning to the trained band; preserve real HFOV in geometry."""
    return float(min(max(float(hfov_rad), MODEL_HFOV_MIN_RAD), MODEL_HFOV_MAX_RAD))


class CoFLSFieldDecoder(nn.Module):
    """Query the visible polar sector and predict a canonical ego-frame field (forward, left). Queries are (theta_tilde, r_tilde); hfov_rad may vary per batch item. Magnitude normalization is stored in v_norm; outputs are not direct actuator speeds."""

    bev_x_max: torch.Tensor
    hfov_rad: torch.Tensor
    v_norm: torch.Tensor

    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 4,
        bev_x_max: float = 5.0,
        hfov_rad: float = math.pi / 2,
        v_norm: float = 5.0,
    ):
        super().__init__()
        if d_model < 4 or d_model % 2:
            raise ValueError("d_model must be even and at least 4")
        if num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.register_buffer("bev_x_max", torch.tensor(float(bev_x_max), dtype=torch.float32))
        self.register_buffer("hfov_rad", torch.tensor(float(hfov_rad), dtype=torch.float32))
        self.register_buffer("v_norm", torch.tensor(float(v_norm), dtype=torch.float32))
        self.set_bev_geometry(bev_x_max=bev_x_max, hfov_rad=hfov_rad, v_norm=v_norm)
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.query_in_dim = 2
        self.coord_dim = 3
        self.vel_dim = 2
        self.pos_proj = nn.Sequential(
            GaussianFourierEncoding(in_features=self.coord_dim, out_features=d_model, scale=10.0),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.cross_attn_layer = nn.ModuleList(
            [
                PreNormDecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=hidden_dim,
                    dropout=dropout,
                    need_self_attn=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.feature_norm = nn.LayerNorm(d_model)
        self.mag_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1), nn.Softplus()
        )
        self.dir_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, self.vel_dim)
        )

    def set_bev_geometry(self, *, bev_x_max: float, hfov_rad: float, v_norm: float) -> None:
        """Update persistent coordinate and field normalization metadata."""
        if not (
            math.isfinite(bev_x_max) and bev_x_max > 0 and math.isfinite(v_norm) and (v_norm > 0)
        ):
            raise ValueError("bev_x_max and v_norm must be finite and positive")
        if not (math.isfinite(hfov_rad) and 0 < hfov_rad <= math.pi):
            raise ValueError("hfov_rad must lie in (0, pi]")
        self.bev_x_max.fill_(float(bev_x_max))
        self.hfov_rad.fill_(float(hfov_rad))
        self.v_norm.fill_(float(v_norm))

    def prepare_context(self, context: torch.Tensor):
        """Project image-language tokens into each decoder layer's key/value space."""
        if context.ndim != 3 or context.shape[-1] != self.d_model:
            raise ValueError("context must have shape [B,T,d_model]")
        return tuple(layer.prepare_context(context) for layer in self.cross_attn_layer)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        hfov_rad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decode(x, self.prepare_context(context), hfov_rad=hfov_rad)

    def decode(self, x: torch.Tensor, query_context, *, hfov_rad: torch.Tensor | None = None):
        """Return [B,N,2] canonical field vectors; hfov_rad accepts a scalar or one value per batch item."""
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError("query must have shape [B,N,2]")
        if len(query_context) != len(self.cross_attn_layer):
            raise ValueError("query context must contain one key/value pair per decoder layer")
        if any(key.shape[0] != x.shape[0] for key, value in query_context):
            raise ValueError("query and context must have the same batch size")
        B = x.shape[0]
        if hfov_rad is None:
            hfov_t = self.hfov_rad
        else:
            hfov_t = torch.as_tensor(hfov_rad, device=x.device, dtype=x.dtype)
        if hfov_t.dim() == 0:
            hfov_bar = hfov_t / (2.0 * THETA_REF_RAD)
        elif hfov_t.dim() == 1:
            if int(hfov_t.shape[0]) != B:
                raise ValueError(f"hfov_rad batch size {int(hfov_t.shape[0])} != x batch {B}")
            hfov_bar = (hfov_t / (2.0 * THETA_REF_RAD)).view(B, 1, 1)
        else:
            raise ValueError(f"hfov_rad must be 0-D or 1-D, got shape {tuple(hfov_t.shape)}")
        theta_bar = x[..., 0:1] * hfov_bar
        x_aug = torch.cat([x, theta_bar], dim=-1)
        local_emb = self.pos_proj(x_aug)
        for layer, layer_context in zip(self.cross_attn_layer, query_context):
            local_emb = layer.decode(local_emb, layer_context)
        local_emb = self.feature_norm(local_emb)
        features = local_emb
        raw_dir = self.dir_head(features)
        dir_unit = F.normalize(raw_dir, dim=-1, eps=1e-08)
        mag = self.mag_head(features)
        v = dir_unit * mag
        return v

    @staticmethod
    def build_sector_query_lattice(
        N_r: int,
        N_theta: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return [N_r*N_theta,2] queries ordered by radius, then angle."""
        N_r = int(N_r)
        N_theta = int(N_theta)
        if N_r <= 0 or N_theta <= 0:
            raise ValueError(f"N_r,N_theta must be >0, got N_r={N_r}, N_theta={N_theta}")
        theta = torch.linspace(-1.0, 1.0, steps=N_theta, device=device, dtype=dtype)
        r = torch.linspace(0.0, 1.0, steps=N_r, device=device, dtype=dtype)
        (theta_g, r_g) = torch.meshgrid(theta, r, indexing="xy")
        return torch.stack([theta_g, r_g], dim=-1).reshape(-1, 2).contiguous()

    def denormalize_to_polar(
        self, canonical: torch.Tensor, hfov_rad: float | None = None
    ) -> torch.Tensor:
        """Convert a normalized query to physical angle (rad) and range (m)."""
        half_fov = float(self.hfov_rad.item()) / 2.0 if hfov_rad is None else float(hfov_rad) / 2.0
        theta_rad = canonical[..., 0:1] * half_fov
        r_m = canonical[..., 1:2] * self.bev_x_max
        return torch.cat([theta_rad, r_m], dim=-1)

    @staticmethod
    def polar_to_cartesian(polar: torch.Tensor) -> torch.Tensor:
        """Convert physical polar coordinates to (forward, left) meters."""
        theta = polar[..., 0]
        r = polar[..., 1]
        x_fwd = r * torch.cos(theta)
        y_lft = r * torch.sin(theta)
        return torch.stack([x_fwd, y_lft], dim=-1)

    def polar_query_to_bev_ego(self, polar_query: torch.Tensor) -> torch.Tensor:
        """Map normalized queries to (forward, left) meters with stored geometry."""
        return self.polar_to_cartesian(self.denormalize_to_polar(polar_query))
