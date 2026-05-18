"""Visualize I-JEPA context/target masks on GeoTIFF crops.

This is a mask visualizer, not a learned-attention visualizer. I-JEPA samples
context and target patch masks before training; these images show whether that
pretext task is hiding and revealing useful geospatial structure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from geo_vlm.data.geotiff_jepa import GeoTiffJEPADataset
from geo_vlm.training.masks import MultiBlockMaskCollator


CONTEXT_COLOR = np.array([0, 210, 255], dtype=np.float32)
TARGET_COLORS = [
    np.array([255, 64, 96], dtype=np.float32),
    np.array([255, 176, 0], dtype=np.float32),
    np.array([140, 92, 255], dtype=np.float32),
    np.array([0, 220, 120], dtype=np.float32),
]
HIDDEN_COLOR = np.array([20, 20, 20], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("~/ijepa_logs/mask_visuals"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional dataset indices to visualize. Defaults to evenly spaced samples.",
    )
    return parser.parse_args()


def _stretch_to_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().numpy().astype(np.float32)
    if array.shape[0] > 3:
        array = array[:3]
    if array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)
    rgb = np.moveaxis(array, 0, -1)

    stretched = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(rgb.shape[-1]):
        values = rgb[..., channel]
        lo, hi = np.percentile(values, [2, 98])
        if hi <= lo:
            stretched[..., channel] = 0.0
        else:
            stretched[..., channel] = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return (stretched * 255.0).astype(np.uint8)


def _indices_to_patch_mask(indices: torch.Tensor, grid_size: int) -> np.ndarray:
    mask = np.zeros((grid_size * grid_size,), dtype=bool)
    mask[indices.detach().cpu().numpy()] = True
    return mask.reshape(grid_size, grid_size)


def _upsample_patch_mask(mask: np.ndarray, patch_size: int) -> np.ndarray:
    return np.repeat(np.repeat(mask, patch_size, axis=0), patch_size, axis=1)


def _blend_where(
    image: np.ndarray,
    mask: np.ndarray,
    color: np.ndarray,
    alpha: float,
) -> np.ndarray:
    output = image.astype(np.float32).copy()
    output[mask] = (1.0 - alpha) * output[mask] + alpha * color
    return np.clip(output, 0, 255).astype(np.uint8)


def _draw_grid(image: np.ndarray, patch_size: int) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    for y in range(0, height + 1, patch_size):
        cv2.line(output, (0, y), (width, y), (255, 255, 255), 1, lineType=cv2.LINE_AA)
    for x in range(0, width + 1, patch_size):
        cv2.line(output, (x, 0), (x, height), (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return output


def _write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _select_indices(dataset_len: int, num_samples: int, requested: list[int] | None) -> list[int]:
    if requested:
        return [idx for idx in requested if 0 <= idx < dataset_len]
    if num_samples >= dataset_len:
        return list(range(dataset_len))
    return np.linspace(0, dataset_len - 1, num_samples, dtype=int).tolist()


def visualize(config: dict, output_dir: Path, num_samples: int, seed: int, indices: list[int] | None) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    crop_size = int(config["data"]["crop_size"])
    patch_size = int(config["model"]["patch_size"])
    grid_size = crop_size // patch_size

    dataset = GeoTiffJEPADataset(
        root=Path(config["data"]["root_path"]),
        image_product=config["data"]["image_product"],
        crop_size=crop_size,
        crop_scale=tuple(config["data"]["crop_scale"]),
        expected_channels=int(config["model"]["in_channels"]),
    )
    collator = MultiBlockMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        enc_mask_scale=tuple(config["mask"]["enc_mask_scale"]),
        pred_mask_scale=tuple(config["mask"]["pred_mask_scale"]),
        aspect_ratio=tuple(config["mask"]["aspect_ratio"]),
        num_enc_masks=int(config["mask"]["num_enc_masks"]),
        num_pred_masks=int(config["mask"]["num_pred_masks"]),
        min_keep=int(config["mask"]["min_keep"]),
        allow_overlap=bool(config["mask"]["allow_overlap"]),
    )

    output_dir = output_dir.expanduser()
    sample_indices = _select_indices(len(dataset), num_samples, indices)
    manifest = {
        "type": "jepa_context_target_mask_visualization",
        "note": "These are sampled I-JEPA masks, not learned attention maps.",
        "root": str(dataset.root),
        "image_product": dataset.image_product,
        "crop_size": crop_size,
        "patch_size": patch_size,
        "seed": seed,
        "samples": [],
    }

    for output_index, dataset_index in enumerate(sample_indices):
        image = dataset[dataset_index]
        images, context_masks, target_masks = collator([image])
        rgb = _stretch_to_uint8(images[0])

        context_patch = _indices_to_patch_mask(context_masks[0][0], grid_size)
        context_pixels = _upsample_patch_mask(context_patch, patch_size)
        target_patches = [_indices_to_patch_mask(mask[0], grid_size) for mask in target_masks]
        target_pixels = [_upsample_patch_mask(mask, patch_size) for mask in target_patches]

        any_target = np.logical_or.reduce(target_pixels)
        visible = context_pixels | any_target
        dimmed = _blend_where(rgb, ~visible, HIDDEN_COLOR, alpha=0.55)

        combined = _blend_where(dimmed, context_pixels, CONTEXT_COLOR, alpha=0.28)
        for target_idx, target_mask in enumerate(target_pixels):
            color = TARGET_COLORS[target_idx % len(TARGET_COLORS)]
            combined = _blend_where(combined, target_mask, color, alpha=0.52)
        combined = _draw_grid(combined, patch_size)

        context_only = _draw_grid(_blend_where(rgb, context_pixels, CONTEXT_COLOR, alpha=0.45), patch_size)
        target_only = rgb.copy()
        for target_idx, target_mask in enumerate(target_pixels):
            color = TARGET_COLORS[target_idx % len(TARGET_COLORS)]
            target_only = _blend_where(target_only, target_mask, color, alpha=0.55)
        target_only = _draw_grid(target_only, patch_size)

        stem = f"sample_{output_index:03d}_idx_{dataset_index:04d}"
        _write_png(output_dir / f"{stem}_input.png", rgb)
        _write_png(output_dir / f"{stem}_combined.png", combined)
        _write_png(output_dir / f"{stem}_context.png", context_only)
        _write_png(output_dir / f"{stem}_targets.png", target_only)

        manifest["samples"].append(
            {
                "dataset_index": dataset_index,
                "source_path": str(dataset.files[dataset_index]),
                "input": f"{stem}_input.png",
                "combined": f"{stem}_combined.png",
                "context": f"{stem}_context.png",
                "targets": f"{stem}_targets.png",
                "context_patch_count": int(context_patch.sum()),
                "target_patch_counts": [int(mask.sum()) for mask in target_patches],
            }
        )

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(sample_indices)} mask visualizations to {output_dir}")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    visualize(config, args.output_dir, args.num_samples, args.seed, args.indices)


if __name__ == "__main__":
    main()
