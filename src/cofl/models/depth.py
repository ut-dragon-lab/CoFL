"""CoFL-S depth patches modulate visual tokens with gated FiLM."""

from __future__ import annotations

import torch
from torch import nn


class DepthPatchAdapter(nn.Module):
    """Embed normalized depth and an optional validity channel into patch tokens."""

    def __init__(self, patch_size: int, out_dim: int, use_mask_channel: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.use_mask_channel = use_mask_channel
        in_ch = 2 if use_mask_channel else 1
        self.proj = nn.Conv2d(
            in_channels=in_ch, out_channels=out_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, depth: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        if depth.dim() != 4 or depth.size(1) != 1:
            raise ValueError(f"depth must be [B,1,H,W], got {tuple(depth.shape)}")
        if self.use_mask_channel:
            if valid_mask is None:
                valid_mask = (torch.isfinite(depth) & (depth > 0) & (depth <= 1)).to(depth.dtype)
            else:
                if valid_mask.dim() != 4 or valid_mask.size(1) != 1:
                    raise ValueError(f"valid_mask must be [B,1,H,W], got {tuple(valid_mask.shape)}")
                valid_mask = valid_mask.float()
            x = torch.cat([depth, valid_mask], dim=1)
        else:
            x = depth
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class DepthFusion(nn.Module):
    """Modulate RGB patch features using depth/scale; input depth is normalized."""

    def __init__(
        self,
        patch_size: int,
        d_model: int,
        depth_hidden_dim: int = 128,
        use_mask_channel: bool = True,
    ):
        super().__init__()
        self.adapter = DepthPatchAdapter(
            patch_size=patch_size, out_dim=depth_hidden_dim, use_mask_channel=use_mask_channel
        )
        self.depth_norm = nn.LayerNorm(depth_hidden_dim)
        film_out_dim = d_model * 2
        self.to_film = nn.Sequential(
            nn.Linear(depth_hidden_dim, film_out_dim // 2),
            nn.GELU(),
            nn.Linear(film_out_dim // 2, film_out_dim),
        )
        self.max_alpha = 0.1
        self.alpha_logit = nn.Parameter(torch.tensor(-2.1972246))
        self.max_shift_gain = 0.05
        self.shift_gain_param = nn.Parameter(torch.tensor(-1.3862944))
        self._init_weights()

    def _init_weights(self):
        for m in self.to_film.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        last_linear = None
        for m in self.to_film.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)

    def effective_alpha(self) -> torch.Tensor:
        return self.max_alpha * torch.sigmoid(self.alpha_logit)

    def effective_shift_gain(self) -> torch.Tensor:
        return self.max_shift_gain * torch.sigmoid(self.shift_gain_param)

    def forward(
        self,
        context: torch.Tensor,
        depth: torch.Tensor,
        depth_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(context, torch.Tensor):
            raise TypeError("context must be a torch.Tensor")
        if context.dim() != 3:
            raise ValueError(f"context must be [B,N,C], got {tuple(context.shape)}")
        if not isinstance(depth, torch.Tensor):
            raise TypeError("depth must be a torch.Tensor")
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth.dim() != 4 or depth.size(1) != 1:
            raise ValueError(f"depth must be [B,1,H,W], got {tuple(depth.shape)}")
        if depth_valid_mask is not None:
            if depth_valid_mask.dim() == 3:
                depth_valid_mask = depth_valid_mask.unsqueeze(1)
            if depth_valid_mask.dim() != 4 or depth_valid_mask.size(1) != 1:
                raise ValueError(
                    f"depth_valid_mask must be [B,1,H,W], got {tuple(depth_valid_mask.shape)}"
                )
        depth_feats = self.adapter(depth, valid_mask=depth_valid_mask)
        if depth_feats.shape[1] != context.shape[1]:
            raise ValueError(
                f"Depth patch count must match context token count. Got depth N={depth_feats.shape[1]} vs context N={context.shape[1]}."
            )
        depth_feats = self.depth_norm(depth_feats)
        film_params = self.to_film(depth_feats)
        (gamma, beta) = film_params.chunk(2, dim=-1)
        alpha = self.effective_alpha().to(dtype=context.dtype)
        shift_gain = self.effective_shift_gain().to(dtype=context.dtype)
        mod_scale = 1.0 + alpha * torch.tanh(gamma)
        mod_shift = shift_gain * torch.tanh(beta)
        out = mod_scale * context + mod_shift
        return out
