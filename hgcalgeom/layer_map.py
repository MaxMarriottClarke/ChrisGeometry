"""Parsers for geometry flat files.

The historical files have evolved over time. The generic reader keeps the
original line around and extracts numeric fields conservatively. The
``parse_chris_geometry`` parser handles the silicon wafer layer layout flat-file
format documented by Chris Seez.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
import re

from .geometry import Point, Wafer

_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

LAYOUT_HEXAGON_WIDTH_MM = 167.4408
DEFAULT_WAFER_SIDE_MM = LAYOUT_HEXAGON_WIDTH_MM / sqrt(3.0)

SILICON_LAYER_TYPES = {
    0: "wafer-centred, sensitive thickness towards vertex",
    1: "wafer-centred, sensitive thickness towards back of HGCAL",
    2: "corner-centred, Y-type",
    3: "corner-centred, lambda-type",
    4: "wafer-centred, rotated by +30 degrees",
}

WAFER_TYPE_NAMES = {
    0: "Full",
    1: "Top",
    2: "Bottom",
    3: "Left",
    4: "Right",
    5: "Five",
    6: "Partial-6",
}


@dataclass(frozen=True, slots=True)
class FlatFileRecord:
    line_number: int
    raw: str
    numbers: tuple[float, ...]
    tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SiliconLayerHeader:
    layer: int
    layer_type: int
    cassette_retractions: tuple[Point, ...]
    line_number: int

    @property
    def layer_type_name(self) -> str:
        return SILICON_LAYER_TYPES.get(self.layer_type, "unknown")


def read_records(path: str | Path) -> list[FlatFileRecord]:
    records: list[FlatFileRecord] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            numbers = tuple(float(x) for x in _NUMBER_RE.findall(stripped))
            records.append(
                FlatFileRecord(
                    line_number=line_number,
                    raw=stripped,
                    numbers=numbers,
                    tokens=tuple(stripped.replace(",", " ").split()),
                )
            )
    return records


def guess_wafers_from_records(records: list[FlatFileRecord], *, wafer_side: float = 1.0) -> list[Wafer]:
    """Build a best-effort wafer list from numeric flat-file records.

    This is intentionally conservative. It assumes the first four numeric
    columns are approximately wafer u, wafer v, x, y. For production use prefer
    ``parse_chris_geometry`` for Chris's silicon flat-file format.
    """

    wafers: list[Wafer] = []
    for record in records:
        if len(record.numbers) < 4:
            continue
        u = int(record.numbers[0])
        v = int(record.numbers[1])
        x = float(record.numbers[2])
        y = float(record.numbers[3])
        wafers.append(
            Wafer(
                u=u,
                v=v,
                center=Point(x, y),
                side=wafer_side,
                file_line=record.line_number,
                metadata={"raw": record.raw, "numbers": record.numbers},
            )
        )
    return wafers


def parse_silicon_headers(path: str | Path) -> list[SiliconLayerHeader]:
    """Parse silicon flat-file header lines.

    Header lines contain the layer number, the layer type, and then cassette
    retraction vectors as x,y pairs. CEE layers have 6 vectors, while CEH layers
    have 12 vectors.
    """

    headers: list[SiliconLayerHeader] = []
    for record in read_records(path):
        tokens = record.tokens
        if len(tokens) < 4:
            continue
        if len(tokens) >= 3 and tokens[2][:1].lower() in {"h", "l"}:
            continue
        try:
            layer = int(tokens[0])
            layer_type = int(tokens[1])
            values = [float(token) for token in tokens[2:]]
        except ValueError:
            continue
        if len(values) % 2 != 0:
            continue
        headers.append(
            SiliconLayerHeader(
                layer=layer,
                layer_type=layer_type,
                cassette_retractions=tuple(Point(values[i], values[i + 1]) for i in range(0, len(values), 2)),
                line_number=record.line_number,
            )
        )
    return headers


def parse_chris_geometry(path: str | Path, *, layer: int | None = None, wafer_side: float | None = None) -> list[Wafer]:
    """Parse Chris Seez's silicon wafer layer layout flat-file.

    Documented data lines have the form::

        layer wafer_type sensor_type x_mm y_mm placement wafer_u wafer_v cassette

    Older Hex dumps may omit the final cassette column; this parser accepts
    both variants and stores ``cassette=None`` when the column is absent.

    The file's first 47 lines are layer headers of the form::

        layer tessellation_flag retraction_vectors...

    tessellation_flag == 1 means the layer is viewed from the back of HGCAL
    (i.e. mirrored), matching HXGHexView's ``seenFromBack`` logic.
    """

    records = read_records(path)

    # Collect per-layer tessellation flags from the header lines (first 47 file
    # lines, identified by tokens[2] not starting with 'h' or 'l').
    layer_types: dict[int, int] = {}
    for record in records:
        tokens = record.tokens
        if len(tokens) < 3:
            continue
        if tokens[2][:1].lower() in {"h", "l"}:
            continue
        try:
            layer_types[int(tokens[0])] = int(tokens[1])
        except ValueError:
            continue

    wafers: list[Wafer] = []
    for record in records:
        tokens = record.tokens
        if len(tokens) < 8:
            continue
        if not tokens[0].lstrip("+-").isdigit() or not tokens[1].lstrip("+-").isdigit():
            continue
        sensor_type = tokens[2].lower()
        if sensor_type[0:1] not in {"h", "l"}:
            continue
        try:
            record_layer = int(tokens[0])
            wafer_type = int(tokens[1])
            x = float(tokens[3])
            y = float(tokens[4])
            placement = int(tokens[5])
            wafer_u = int(tokens[6])
            wafer_v = int(tokens[7])
            cassette = int(tokens[8]) if len(tokens) >= 9 else None
        except ValueError:
            continue
        if layer is not None and record_layer != layer:
            continue

        side = wafer_side if wafer_side is not None else DEFAULT_WAFER_SIDE_MM
        wafers.append(
            Wafer(
                u=wafer_u,
                v=wafer_v,
                center=Point(x, y),
                side=side,
                is_ld=sensor_type.startswith("l"),
                is_partial=wafer_type != 0,
                partial_type=wafer_type,
                placement=placement,
                seen_from_back=(layer_types.get(record_layer, 0) == 1),
                cassette=cassette,
                file_line=record.line_number,
                metadata={
                    "raw": record.raw,
                    "layer": record_layer,
                    "sensor_type": sensor_type,
                    "wafer_type": wafer_type,
                    "wafer_type_name": WAFER_TYPE_NAMES.get(wafer_type, "unknown"),
                },
            )
        )
    return wafers
