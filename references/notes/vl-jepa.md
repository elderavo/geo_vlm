# VL-JEPA

## Summary

VL-JEPA extends the JEPA idea into vision-language modeling by predicting target text embeddings rather than autoregressively generating tokens. The paper's central move is still relevant to us: learn in a continuous semantic space first, and decode into a task-specific output only when needed. It is not, however, the original I-JEPA image-pretraining paper that should guide our first implementation.

## What we borrow

- The clean separation between:
  - an encoder that produces reusable representations,
  - a predictor/decoder that maps those representations into task-specific outputs,
  - and a later decoding step that turns those outputs into user-facing artifacts.
- The general JEPA principle that prediction in representation space can be preferable to direct reconstruction in data space.
- The design habit of keeping task-specific decoding lightweight relative to the backbone.

## What we do not borrow

- We do not need text targets, token decoders, or vision-language alignment for the current road-extraction MVP.
- We should not use this paper as the implementation blueprint for image-only JEPA pretraining on SpaceNet.
- Its claims about selective text decoding do not transfer directly to road masks or GeoSPARQL export.

## Most relevant implementation notes for `geo-vlm`

1. **Keep the backbone and the task head separate.**
   - This supports our current plan: one shared encoder, multiple downstream geospatial heads.

2. **Predict structured task outputs before serialization.**
   - The paper reinforces the broader point that representations should feed task outputs, not force the model to emit surface forms directly.
   - For us, that means:
     - latent image features -> road mask / centerline,
     - then deterministic vectorization,
     - then deterministic WKT/RDF export.

3. **Treat decoders as task-specific, not universal.**
   - A road decoder and a building decoder can share an encoder while still having different output geometry and losses.

4. **Use the paper as conceptual support, not as code guidance.**
   - For actual implementation details such as image masking, target blocks, teacher encoder updates, and JEPA loss construction, we still need the original I-JEPA source.

## Why it matters for `geo-vlm`

- It supports the architectural choice to separate reusable latent representation learning from downstream decoders and final artifact generation.

## Open questions

- Once we add the original I-JEPA paper, which pieces of the image-only training recipe should we reproduce exactly and which should we adapt for overhead imagery?
