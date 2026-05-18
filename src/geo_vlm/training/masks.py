"""Multi-block masking adapted from the official I-JEPA implementation."""

from __future__ import annotations

import math
from multiprocessing import Value

import torch


class MultiBlockMaskCollator:
    def __init__(
        self,
        *,
        input_size: int,
        patch_size: int,
        enc_mask_scale: tuple[float, float],
        pred_mask_scale: tuple[float, float],
        aspect_ratio: tuple[float, float],
        num_enc_masks: int,
        num_pred_masks: int,
        min_keep: int,
        allow_overlap: bool,
    ) -> None:
        self.height = input_size // patch_size
        self.width = input_size // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.num_enc_masks = num_enc_masks
        self.num_pred_masks = num_pred_masks
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self._itr_counter = Value("i", -1)

    def step(self) -> int:
        with self._itr_counter.get_lock():
            self._itr_counter.value += 1
            return self._itr_counter.value

    def _sample_block_size(
        self,
        *,
        generator: torch.Generator,
        scale: tuple[float, float],
        aspect_ratio_scale: tuple[float, float],
    ) -> tuple[int, int]:
        rand = torch.rand(1, generator=generator).item()
        mask_scale = scale[0] + rand * (scale[1] - scale[0])
        max_keep = int(self.height * self.width * mask_scale)
        aspect_ratio = aspect_ratio_scale[0] + rand * (
            aspect_ratio_scale[1] - aspect_ratio_scale[0]
        )
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1
        return h, w

    def _sample_block_mask(
        self,
        block_size: tuple[int, int],
        acceptable_regions: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = block_size
        tries = 0
        timeout = original_timeout = 20
        while True:
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top : top + h, left : left + w] = 1
            if acceptable_regions is not None:
                region_count = max(int(len(acceptable_regions) - tries), 0)
                for region in acceptable_regions[:region_count]:
                    mask *= region
            flat_mask = torch.nonzero(mask.flatten()).squeeze()
            if flat_mask.numel() > self.min_keep:
                complement = torch.ones((self.height, self.width), dtype=torch.int32)
                complement[top : top + h, left : left + w] = 0
                return flat_mask, complement
            timeout -= 1
            if timeout == 0:
                tries += 1
                timeout = original_timeout

    def __call__(
        self, batch: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        images = torch.utils.data.default_collate(batch)
        generator = torch.Generator().manual_seed(self.step())
        pred_size = self._sample_block_size(
            generator=generator,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )
        enc_size = self._sample_block_size(
            generator=generator,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1.0, 1.0),
        )

        pred_masks_per_image: list[list[torch.Tensor]] = []
        enc_masks_per_image: list[list[torch.Tensor]] = []
        min_pred = self.height * self.width
        min_enc = self.height * self.width

        for _ in range(len(batch)):
            pred_masks = []
            complements = []
            for _ in range(self.num_pred_masks):
                mask, complement = self._sample_block_mask(pred_size)
                pred_masks.append(mask)
                complements.append(complement)
                min_pred = min(min_pred, len(mask))
            pred_masks_per_image.append(pred_masks)

            acceptable = None if self.allow_overlap else complements
            enc_masks = []
            for _ in range(self.num_enc_masks):
                mask, _ = self._sample_block_mask(enc_size, acceptable)
                enc_masks.append(mask)
                min_enc = min(min_enc, len(mask))
            enc_masks_per_image.append(enc_masks)

        pred_masks_per_image = [
            [mask[:min_pred] for mask in masks] for masks in pred_masks_per_image
        ]
        enc_masks_per_image = [
            [mask[:min_enc] for mask in masks] for masks in enc_masks_per_image
        ]
        pred_masks = torch.utils.data.default_collate(pred_masks_per_image)
        enc_masks = torch.utils.data.default_collate(enc_masks_per_image)
        return images, list(enc_masks), list(pred_masks)
