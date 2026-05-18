# I-JEPA

## Summary

I-JEPA trains an image encoder through self-supervised prediction in latent space rather than reconstructing pixels directly. The method is designed to learn semantically useful image representations without labels. For `geo-vlm`, the relevant idea is to pretrain a reusable encoder before fitting task-specific geospatial decoder heads.

## What we borrow

- Self-supervised encoder pretraining on unlabeled imagery.
- The separation between representation learning and downstream supervised tasks.
- The expectation that a strong encoder can support multiple later heads.

## What we do not borrow

- We are not yet committing to a faithful reproduction of the full published architecture.
- We are not assuming that JEPA alone solves road extraction without a supervised decoder.

## Why it matters for `geo-vlm`

- It motivates the shared backbone used by road, building, and future object-specific decoders.

## Open questions

- Which JEPA variant is practical for multispectral or RGB SpaceNet imagery?
- What label-efficiency gains do we actually observe on the road task?
