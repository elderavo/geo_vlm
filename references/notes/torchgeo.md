# TorchGeo

## Summary

TorchGeo provides geospatial datasets, samplers, and transforms for PyTorch workflows. It includes SpaceNet support and gives us a tested dataset abstraction instead of requiring hand-written geospatial IO from the start. For `geo-vlm`, it is the preferred boundary between raw benchmark data and model training code.

## What we borrow

- TorchGeo dataset wrappers for SpaceNet.
- A library-owned adapter layer for raster/vector dataset handling.

## What we do not borrow

- We should not assume TorchGeo removes all alignment, evaluation, or modeling work from the pipeline.

## Why it matters for `geo-vlm`

- It lets us start from a cleaner, more reproducible dataset interface while we focus on model design.

## Open questions

- Do we need custom sampling or transforms beyond the built-in dataset support?
