from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from geo_vlm.data.geotiff_jepa import GeoTiffJEPADataset, inspect_rasters


def _write_tif(path: Path) -> None:
    data = np.stack(
        [
            np.full((1300, 1300), 1000, dtype=np.uint16),
            np.full((1300, 1300), 2000, dtype=np.uint16),
            np.full((1300, 1300), 3000, dtype=np.uint16),
        ]
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=1300,
        width=1300,
        count=3,
        dtype="uint16",
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(data)


def test_ps_rgb_dataset_loads_three_channel_crops(tmp_path: Path) -> None:
    image = tmp_path / "SN3_roads_train_AOI_5_Khartoum_PS-RGB_img1.tif"
    _write_tif(image)

    dataset = GeoTiffJEPADataset(
        root=tmp_path,
        image_product="PS-RGB",
        crop_size=224,
        crop_scale=(0.3, 1.0),
        expected_channels=3,
    )

    sample = dataset[0]
    assert sample.shape == (3, 224, 224)
    assert sample.dtype.is_floating_point


def test_raster_inspection_records_shape_and_range(tmp_path: Path) -> None:
    image = tmp_path / "SN3_roads_train_AOI_5_Khartoum_PS-RGB_img1.tif"
    _write_tif(image)

    metadata = inspect_rasters([image])

    assert metadata[0].count == 3
    assert metadata[0].width == 1300
    assert metadata[0].height == 1300
    assert metadata[0].min_values == (1000.0, 2000.0, 3000.0)
    assert metadata[0].max_values == (1000.0, 2000.0, 3000.0)
