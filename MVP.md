# MVP: JEPA-to-GeoSPARQL Road Extraction

## One-sentence description

We pretrain a JEPA encoder on SpaceNet imagery, train a lightweight supervised decoder to convert the learned image representations into road masks, and then vectorize those road predictions into GeoSPARQL-compatible WKT geometries.

## What the MVP does

1. **Learn image representations**
   - Use the SpaceNet road imagery as input to a self-supervised JEPA encoder.
   - The JEPA stage learns useful latent structure from the imagery without requiring road labels.

2. **Predict roads from the learned representation**
   - Use the official SpaceNet road labels to train a small supervised decoder head on top of the JEPA features.
   - The decoder outputs a road mask or road centerline mask for each image tile.

3. **Convert predictions into geospatial knowledge**
   - Post-process the predicted masks into vector road geometries.
   - Serialize those vectors as GeoSPARQL-compatible Well-Known Text (WKT), such as `LINESTRING (...)`, inside RDF triples.

## MVP pipeline

```text
SpaceNet imagery
  -> JEPA encoder
  -> latent image features
  -> supervised road-mask decoder
  -> predicted road mask / centerline
  -> vectorization
  -> GeoSPARQL WKT output
```

## What this proves

This MVP demonstrates that:

- self-supervised geospatial representations can be learned from satellite imagery,
- those representations can support supervised road extraction,
- and neural predictions can be converted into standards-based geospatial outputs that are queryable outside the model.

## Important implementation boundary

The model does **not** directly generate RDF or WKT strings. That would mix perception with serialization and make the system harder to debug. Instead:

- the neural model predicts road structure,
- a deterministic post-processing stage converts that structure into vectors,
- and a deterministic exporter writes GeoSPARQL/RDF.

This separation keeps the system easier to evaluate, explain, and extend.
