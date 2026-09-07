"""CoFL-S action-query head and public action IDs."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import PreNormDecoderBlock

ACTION_STOP_ID = 0
ACTION_MOVE_FORWARD_ID = 1
ACTION_TURN_LEFT_ID = 2
ACTION_TURN_RIGHT_ID = 3
ACTION_IGNORE_INDEX = -100


class NavigationActionHead(nn.Module):
    """Four action queries predict STOP, MOVE_FORWARD, TURN_LEFT and TURN_RIGHT logits."""

    def __init__(
        self,
        d_model: int = 768,
        num_actions: int = 4,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_actions = int(num_actions)
        self.action_queries = nn.Parameter(
            torch.randn(self.num_actions, d_model) * d_model ** (-0.5)
        )
        self.query_pos_mlp = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.layers = nn.ModuleList(
            [
                PreNormDecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=4 * d_model,
                    dropout=dropout,
                    need_self_attn=False,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.logit_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
        )

    def forward(
        self, context: torch.Tensor, context_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if context.dim() != 3:
            raise ValueError(f"context must be [B, N_ctx, D], got {tuple(context.shape)}")
        batch_size = int(context.shape[0])
        queries = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries + self.query_pos_mlp(self.action_queries).unsqueeze(0)
        for layer in self.layers:
            queries = layer(queries, context, key_padding_mask=context_padding_mask)
        queries = self.final_norm(queries)
        logits = self.logit_head(queries).squeeze(-1)
        return logits

    @staticmethod
    def get_loss(
        logits: torch.Tensor, action_id: torch.Tensor, *, label_smoothing: float = 0.0
    ) -> torch.Tensor:
        target = action_id.long().view(-1)
        if int((target != ACTION_IGNORE_INDEX).sum().item()) == 0:
            return logits.sum() * 0.0
        return F.cross_entropy(
            logits, target, ignore_index=ACTION_IGNORE_INDEX, label_smoothing=float(label_smoothing)
        )

    @staticmethod
    def stop_probability(logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=-1)[..., ACTION_STOP_ID]

    def all_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=-1)
