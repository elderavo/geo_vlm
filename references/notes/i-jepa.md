# I-JEPA

## Summary

I-JEPA learns image representations by predicting the **latent representations** of masked target regions from a visible context region in the same image. It is deliberately non-generative: the model predicts in embedding space instead of reconstructing pixels. The paper's central claim is that this produces more semantic representations when the masking setup forces the model to reason about meaningful image structure.

## Core training recipe from the paper

1. Split each image into non-overlapping patches.
2. Pass the full image through a **target encoder** to obtain patch-level target representations.
3. Sample several target blocks from the target representations.
4. Sample one large context block from the image and remove regions overlapping the targets.
5. Pass only the visible context patches through the **context encoder**.
6. Use a **predictor** conditioned on positional mask tokens to predict the target-block representations.
7. Optimize average patch-level L2 loss between predicted and target representations.
8. Update the target encoder with an exponential moving average of the context encoder weights.

## The three modules that matter

- **Context encoder**
  - trainable image encoder
  - receives only visible context patches

- **Target encoder**
  - produces the latent targets
  - updated by EMA, not direct gradient descent

- **Predictor**
  - maps context features plus target-position tokens into predicted target features
  - narrower than the main encoder in the original paper

## Masking details worth preserving

The paper is unusually explicit that the masking scheme matters.

- They typically sample **4 target blocks** per image.
- Target blocks use:
  - scale range: **0.15 to 0.20**
  - aspect-ratio range: **0.75 to 1.5**
- The context block uses:
  - scale range: **0.85 to 1.0**
  - aspect ratio: **1.0**
- Overlap between context and target blocks is removed from the context.

The rationale is important:

- target blocks must be large enough to require semantic prediction,
- context must be informative enough to support that prediction,
- and the task must remain non-trivial by hiding the actual target region.

## What we should borrow for `geo-vlm`

- **Predict representations, not pixels.**
  - This is the whole reason to use JEPA rather than a masked autoencoder baseline.

- **Keep a separate target encoder updated by EMA.**
  - The paper treats this as essential for stable training with ViTs.

- **Use multiple target blocks and a spatially broad context.**
  - This is likely relevant for roads because road identity is often inferred from continuation, junctions, and surrounding structure rather than a single local texture patch.

- **Treat mask geometry as a first-class hyperparameter.**
  - For overhead imagery, block scale may need tuning because roads are thin, elongated, and often span long distances.

- **Freeze or partially freeze the pretrained encoder for early downstream experiments.**
  - This gives us a clean first test of whether the representation is useful before adding end-to-end fine-tuning complexity.

## Paper versus released code

- The paper describes an L2 latent prediction loss.
- The released Meta implementation uses `smooth_l1_loss` in its training loop.
- For the first parity-oriented implementation in `geo-vlm`, follow the released code so our behavior matches the reference implementation we have already smoke-tested. Revisit pure L2 only as a deliberate ablation later.

## What may need adaptation for overhead imagery

- **Roads are long and thin.**
  - Square-ish target blocks from natural-image pretraining may not be ideal for geospatial structure.
  - We may eventually test elongated target masks or multiscale targets.

- **Semantics differ from ImageNet.**
  - In overhead imagery, useful structure may be topological and spatial rather than object-centric.

- **Tile boundaries matter.**
  - If roads are cut by tile edges, the context/target sampling strategy may accidentally teach the model partial-road habits.

- **RGB versus multispectral inputs.**
  - The paper is RGB/ImageNet-focused. We need to decide whether our first pass uses RGB only or all available bands.

## How this informs our implementation order

1. Build a patch sampler that can reproduce the paper's context/target block strategy.
2. Implement:
   - context encoder,
   - EMA target encoder,
   - predictor,
   - L2 latent prediction loss.
3. Train on unlabeled SpaceNet imagery.
4. Freeze the context encoder and train a road decoder head.
5. Only after the basic system works, experiment with road-specific mask geometries.

## What we do not borrow blindly

- We should not assume the ImageNet masking ranges are optimal for satellite imagery.
- We should not assume stronger downstream performance without running the comparison against simpler baselines.
- We should not confuse JEPA pretraining with the supervised road decoder; JEPA learns reusable image features, not road masks by itself.

## Why it matters for `geo-vlm`

I-JEPA gives us the concrete image-only training recipe that supports our architecture:

```text
unlabeled imagery
  -> JEPA pretraining
  -> reusable encoder
  -> road decoder / future building decoder
```

It is the source we should use when implementing `pretrain_jepa.py`.

## Open questions

- Do the original square-ish target masks work well enough for roads, or do elongated masks help?
- Should our first baseline use RGB imagery only for simplicity, then expand to multispectral?
- How much label efficiency do we actually gain on SpaceNet road extraction versus training the decoder from scratch?
