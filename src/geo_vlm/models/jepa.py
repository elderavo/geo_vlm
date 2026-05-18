"""I-JEPA model components adapted for configurable input channels."""

from __future__ import annotations

import copy
import math
from functools import partial

import numpy as np
import torch
from torch import nn


def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, std=std)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def apply_masks(x: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    all_x = []
    for mask in masks:
        gather_index = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=gather_index))
    return torch.cat(all_x, dim=0)


def repeat_interleave_batch(x: torch.Tensor, batch_size: int, repeat: int) -> torch.Tensor:
    groups = len(x) // batch_size
    return torch.cat(
        [
            torch.cat([x[i * batch_size : (i + 1) * batch_size] for _ in range(repeat)], dim=0)
            for i in range(groups)
        ],
        dim=0,
    )


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + y
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.num_patches = (img_size // patch_size) ** 2
        pos = get_2d_sincos_pos_embed(embed_dim, int(math.sqrt(self.num_patches)))
        self.register_buffer("pos_embed", torch.from_numpy(pos).float().unsqueeze(0), persistent=False)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor, masks: list[torch.Tensor] | None = None) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed.to(x.device)
        if masks is not None:
            x = apply_masks(x, masks)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class Predictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches: int,
        embed_dim: int,
        predictor_embed_dim: int,
        depth: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        pos = get_2d_sincos_pos_embed(predictor_embed_dim, int(math.sqrt(num_patches)))
        self.register_buffer("pos_embed", torch.from_numpy(pos).float().unsqueeze(0), persistent=False)
        self.blocks = nn.ModuleList([Block(predictor_embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(predictor_embed_dim, eps=1e-6)
        self.proj = nn.Linear(predictor_embed_dim, embed_dim)
        trunc_normal_(self.mask_token)

    def forward(
        self,
        x: torch.Tensor,
        context_masks: list[torch.Tensor],
        target_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = len(x) // len(context_masks)
        x = self.embed(x)
        x = x + apply_masks(self.pos_embed.repeat(batch_size, 1, 1), context_masks)
        context_tokens = x.size(1)

        pos_embs = apply_masks(self.pos_embed.repeat(batch_size, 1, 1), target_masks)
        pos_embs = repeat_interleave_batch(pos_embs, batch_size, repeat=len(context_masks))
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1) + pos_embs
        x = torch.cat([x.repeat(len(target_masks), 1, 1), pred_tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.proj(self.norm(x[:, context_tokens:]))


MODEL_SPECS = {
    "vit_tiny": dict(embed_dim=192, depth=12, num_heads=3),
    "vit_small": dict(embed_dim=384, depth=12, num_heads=6),
    "vit_base": dict(embed_dim=768, depth=12, num_heads=12),
    "vit_large": dict(embed_dim=1024, depth=24, num_heads=16),
    "vit_huge": dict(embed_dim=1280, depth=32, num_heads=16),
}


def build_jepa_models(
    *,
    model_name: str,
    img_size: int,
    patch_size: int,
    in_channels: int,
    predictor_depth: int,
    predictor_embed_dim: int,
) -> tuple[VisionTransformer, VisionTransformer, Predictor]:
    spec = MODEL_SPECS[model_name]
    encoder = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        **spec,
    )
    target_encoder = copy.deepcopy(encoder)
    predictor = Predictor(
        num_patches=encoder.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=predictor_embed_dim,
        depth=predictor_depth,
        num_heads=encoder.num_heads,
    )
    for parameter in target_encoder.parameters():
        parameter.requires_grad = False
    return encoder, target_encoder, predictor
