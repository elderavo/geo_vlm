"""Export road vectors into GeoSPARQL-compatible RDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import XSD


GEO = Namespace("http://www.opengis.net/ont/geosparql#")
EX = Namespace("https://example.org/geo-vlm/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def export_segments(input_path: Path, output_path: Path) -> None:
    """Serialize road segments to Turtle.

    RDF export is intentionally deterministic. The neural model should predict
    geometry earlier in the pipeline; this stage should only translate domain
    objects into a standard interchange format.
    """

    segments = gpd.read_file(input_path)

    graph = Graph()
    graph.bind("geo", GEO)
    graph.bind("ex", EX)

    for index, row in segments.iterrows():
        segment_uri = URIRef(EX[f"road_segment/{index}"])
        geometry_uri = URIRef(EX[f"geometry/{index}"])

        graph.add((segment_uri, RDF.type, GEO.Feature))
        graph.add((segment_uri, GEO.hasGeometry, geometry_uri))
        graph.add((geometry_uri, RDF.type, GEO.Geometry))
        graph.add(
            (
                geometry_uri,
                GEO.asWKT,
                Literal(row.geometry.wkt, datatype=GEO.wktLiteral),
            )
        )

        if "confidence" in row and row["confidence"] is not None:
            graph.add(
                (
                    segment_uri,
                    EX.confidence,
                    Literal(float(row["confidence"]), datatype=XSD.decimal),
                )
            )

    graph.serialize(destination=output_path, format="turtle")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.input_dir / "road_segments.geojson"
    output_path = args.output_dir / "road_segments.ttl"

    export_segments(input_path, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
