"""Natural-language request normalization for interactive inference.

The parser intentionally keeps feature names as free text. It only normalizes
output constraints and structured vector-product requirements, so the model
path remains independent of feature categories and task-specific branches.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class InteractionRequest:
    raw_text: str
    normalized_text: str
    output_format: str
    export_format: Optional[str]
    expects_structured_vector: bool


_STRUCTURED_VECTOR_HINTS = (
    "[det]",
    "[vg]",
    "vector",
    "extract",
    "extraction",
    "coordinate",
    "coordinates",
    "geometry",
    "geometries",
    "boundary",
    "boundaries",
    "line",
    "lines",
    "polygon",
    "polygons",
    "geojson",
    "featurecollection",
    "feature collection",
    "shapefile",
    "shape file",
    "shp",
    "geopackage",
    "gpkg",
    "wkt",
)

_TEXT_TASK_HINTS = (
    "describe",
    "caption",
    "classify",
    "classification",
    "scene",
    "question",
    "answer",
    "vqa",
    "visual question",
    "chat",
    "explain",
    "summarize",
)

_GEOJSON_HINTS = (
    "geojson",
    "featurecollection",
    "feature collection",
)

_JSON_HINTS = (
    "json",
    "return json",
    "json only",
)

_WKT_HINTS = (
    "wkt",
    "well-known text",
)

_EXPORT_FORMAT_HINTS = {
    "shapefile": ("shapefile", "shape file", ".shp", " shp"),
    "gpkg": ("geopackage", "gpkg", ".gpkg"),
}


def _contains_any(text: str, markers) -> bool:
    return any(marker in text for marker in markers)


def _resolve_export_format(text_lower: str) -> Optional[str]:
    for name, markers in _EXPORT_FORMAT_HINTS.items():
        if _contains_any(text_lower, markers):
            return name
    return None


def _resolve_output_format(text_lower: str, default_format: str) -> str:
    requested = str(default_format or "auto").strip().lower()
    if requested not in {"auto", "text", "json", "geojson", "wkt"}:
        requested = "auto"
    if requested != "auto":
        return requested
    if _contains_any(text_lower, _GEOJSON_HINTS):
        return "geojson"
    if _contains_any(text_lower, _WKT_HINTS):
        return "wkt"
    if _resolve_export_format(text_lower) is not None:
        return "geojson"
    if _contains_any(text_lower, _STRUCTURED_VECTOR_HINTS):
        return "geojson"
    if _contains_any(text_lower, _JSON_HINTS):
        return "json"
    if _contains_any(text_lower, _TEXT_TASK_HINTS):
        return "text"
    return "text"


def _build_output_contract(output_format: str, export_format: Optional[str], json_only: bool) -> str:
    if output_format == "geojson":
        parts = [
            "Return the requested vector objects as one valid GeoJSON FeatureCollection.",
            "Use continuous normalized image coordinates in [0, 1] unless georeference metadata is available.",
            "Treat the user's requested feature scope as an open vocabulary query, not a fixed class list.",
            "Return only objects that match the user's request.",
            "Do not force background or non-requested regions into any known category.",
            "Keep ambiguous objects unassigned unless the user explicitly asks for uncertain candidates.",
        ]
        if export_format:
            parts.append(
                f"The result must be suitable for export to {export_format}; return GeoJSON in this response."
            )
        if json_only:
            parts.append("Return JSON only.")
        return " ".join(parts)
    if output_format == "json":
        return "Return one valid JSON object. Return JSON only." if json_only else "Return one valid JSON object."
    if output_format == "wkt":
        return "Return the requested vector objects as WKT geometries with one geometry per line."
    return "Answer the user's request directly and keep the response grounded in the provided image."


def normalize_interaction_instruction(
    text: str,
    default_output_format: str = "auto",
    json_only: bool = True,
) -> InteractionRequest:
    raw = str(text or "").strip()
    if not raw:
        return InteractionRequest(
            raw_text=raw,
            normalized_text=raw,
            output_format="text",
            export_format=None,
            expects_structured_vector=False,
        )

    lower = raw.lower()
    export_format = _resolve_export_format(lower)
    output_format = _resolve_output_format(lower, default_output_format)
    expects_structured_vector = output_format in {"geojson", "wkt"} or export_format is not None

    contract = _build_output_contract(output_format, export_format, json_only)
    if contract.lower() in lower:
        normalized = raw
    else:
        normalized = f"{raw}\n\nOutput contract: {contract}"

    return InteractionRequest(
        raw_text=raw,
        normalized_text=normalized,
        output_format=output_format,
        export_format=export_format,
        expects_structured_vector=expects_structured_vector,
    )
