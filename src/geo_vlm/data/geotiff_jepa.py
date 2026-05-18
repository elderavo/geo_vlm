"""GeoTIFF dataset utilities for JEPA pretraining."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class RasterMetadata:
    path: str
    width: int
    height: int
    count: int
    dtypes: tuple[str, ...]
    crs: str | None
    min_values: tuple[float, ...]
    max_values: tuple[float, ...]


def discover_product_files(root: Path, image_product: str) -> list[Path]:
    pattern = f"*_{image_product}_*.tif"
    return sorted(root.expanduser().rglob(pattern))


def inspect_rasters(paths: list[Path], max_files: int | None = None) -> list[RasterMetadata]:
    selected = paths if max_files is None else paths[:max_files]
    metadata: list[RasterMetadata] = []
    for path in selected:
        with rasterio.open(path) as src:
            band_mins: list[float] = []
            band_maxes: list[float] = []
            for band_idx in range(1, src.count + 1):
                band = src.read(band_idx)
                band_mins.append(float(np.nanmin(band)))
                band_maxes.append(float(np.nanmax(band)))
            metadata.append(
                RasterMetadata(
                    path=str(path),
                    width=src.width,
                    height=src.height,
                    count=src.count,
                    dtypes=tuple(src.dtypes),
                    crs=None if src.crs is None else str(src.crs),
                    min_values=tuple(band_mins),
                    max_values=tuple(band_maxes),
                )
            )
    return metadata


def write_manifest(
    *,
    output_path: Path,
    root: Path,
    image_product: str,
    files: list[Path],
    metadata: list[RasterMetadata],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "root": str(root.expanduser()),
        "image_product": image_product,
        "file_count": len(files),
        "shape_contract": "[channels, crop_size, crop_size]",
        "samples": [item.__dict__ for item in metadata],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class GeoTiffJEPADataset(Dataset[torch.Tensor]):
    """Load RGB GeoTIFF tiles and sample random crops for JEPA pretraining."""

    def __init__(
        self,
        *,
        root: Path,
        image_product: str,
        crop_size: int,
        crop_scale: tuple[float, float],
        expected_channels: int,
    ) -> None:
        self.root = root.expanduser()
        self.image_product = image_product
        self.crop_size = crop_size
        self.crop_scale = crop_scale
        self.expected_channels = expected_channels
        self.files = discover_product_files(self.root, image_product)
        if not self.files:
            raise FileNotFoundError(
                f"No TIFFs found for product {image_product!r} under {self.root}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.files[index]
        with rasterio.open(path) as src:
            image = src.read()

        if image.shape[0] != self.expected_channels:
            raise ValueError(
                f"{path} has {image.shape[0]} channels; expected {self.expected_channels}"
            )

        tensor = torch.from_numpy(image.astype(np.float32))
        if np.issubdtype(image.dtype, np.integer):
            dtype_max = float(np.iinfo(image.dtype).max)
            tensor = tensor / dtype_max

        i, j, h, w = RandomResizedCrop.get_params(
            tensor,
            scale=self.crop_scale,
            ratio=(1.0, 1.0),
        )
        return TF.resized_crop(
            tensor,
            top=i,
            left=j,
            height=h,
            width=w,
            size=[self.crop_size, self.crop_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
