# GeoSPARQL

## Summary

GeoSPARQL is an OGC standard for representing and querying geospatial information in RDF. It defines geospatial features, geometries, and literal forms such as WKT. For `geo-vlm`, it provides the symbolic output contract after neural predictions have already been converted into vector geometry.

## What we borrow

- `geo:Feature`
- `geo:Geometry`
- `geo:hasGeometry`
- `geo:asWKT`

## What we do not borrow

- We do not ask the neural model to emit RDF syntax directly.

## Why it matters for `geo-vlm`

- It gives downstream users a standards-based, queryable representation of predicted roads and later buildings.

## Open questions

- Which application-specific classes and predicates should sit beside the GeoSPARQL core vocabulary?
