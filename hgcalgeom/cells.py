"""Silicon cell generation for LD and HD wafers.

This module builds regular grid-cell geometry for full wafers and clips those
cells to wafer or partial-wafer polygons. The partial polygons use the Zoltan
dicing-line points from Chris Seez's Mac application, transformed to the local
wafer frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt

from .geometry import Point, Wafer, polygon_area
from .interface import default_cell_set
from .layer_map import DEFAULT_WAFER_SIDE_MM


DENSITY_LD = 8
DENSITY_HD = 12
MIN_POLYGON_AREA_MM2 = 1.0e-6


def _rotate_offset(dx: float, dy: float, angle_rad: float) -> tuple[float, float]:
    c = cos(angle_rad)
    s = sin(angle_rad)
    return c * dx - s * dy, s * dx + c * dy


def _flat_top_hex_offsets(side: float, angle_rad: float = 0.0) -> list[tuple[float, float]]:
    """Return flat-top hexagon offsets, optionally rotated."""

    h = sqrt(0.75) * side
    offsets = [
        (side, 0.0),
        (0.5 * side, h),
        (-0.5 * side, h),
        (-side, 0.0),
        (-0.5 * side, -h),
        (0.5 * side, -h),
    ]
    if angle_rad == 0.0:
        return offsets
    return [_rotate_offset(dx, dy, angle_rad) for dx, dy in offsets]


def _signed_area(points: list[Point]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for a, b in zip(points, points[1:] + points[:1]):
        total += a.x * b.y - b.x * a.y
    return 0.5 * total


def _ensure_ccw(points: list[Point]) -> list[Point]:
    return points if _signed_area(points) >= 0.0 else list(reversed(points))


def _intersection(a: Point, b: Point, p: Point, q: Point) -> Point:
    ax = b.x - a.x
    ay = b.y - a.y
    bx = q.x - p.x
    by = q.y - p.y
    denom = ax * by - ay * bx
    if abs(denom) < 1.0e-12:
        return b
    t = ((p.x - a.x) * by - (p.y - a.y) * bx) / denom
    return Point(a.x + t * ax, a.y + t * ay)


def _inside_half_plane(point: Point, edge_start: Point, edge_end: Point) -> bool:
    return (edge_end.x - edge_start.x) * (point.y - edge_start.y) - (edge_end.y - edge_start.y) * (
        point.x - edge_start.x
    ) >= -1.0e-9


def clip_polygon(subject: list[Point], clip: list[Point]) -> list[Point]:
    """Clip a polygon by a convex polygon using Sutherland-Hodgman."""

    output = _ensure_ccw(subject)
    clip_ccw = _ensure_ccw(clip)
    for edge_start, edge_end in zip(clip_ccw, clip_ccw[1:] + clip_ccw[:1]):
        if not output:
            break
        input_points = output
        output = []
        previous = input_points[-1]
        previous_inside = _inside_half_plane(previous, edge_start, edge_end)
        for current in input_points:
            current_inside = _inside_half_plane(current, edge_start, edge_end)
            if current_inside:
                if not previous_inside:
                    output.append(_intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif previous_inside:
                output.append(_intersection(previous, current, edge_start, edge_end))
            previous = current
            previous_inside = current_inside
    return output


@dataclass(frozen=True, slots=True)
class SiliconCell:
    """Drawable silicon cell polygon in layer-global coordinates."""

    layer: int | None
    wafer_u: int
    wafer_v: int
    iu: int
    iv: int
    center: Point
    side: float
    is_ld: bool
    wafer_type: int
    sensor_type: str | None = None
    orientation_rad: float = 0.0
    vertices: tuple[Point, ...] | None = None

    def corners(self) -> list[Point]:
        if self.vertices is not None:
            return list(self.vertices)
        return [
            Point(self.center.x + dx, self.center.y + dy)
            for dx, dy in _flat_top_hex_offsets(self.side, self.orientation_rad)
        ]


@dataclass(frozen=True, slots=True)
class LocalSiliconCell:
    iu: int
    iv: int
    center: Point
    side: float

    def corners(self) -> list[Point]:
        return [Point(self.center.x + dx, self.center.y + dy) for dx, dy in _flat_top_hex_offsets(self.side)]


def density_number(*, is_ld: bool) -> int:
    return DENSITY_LD if is_ld else DENSITY_HD


def regular_cell_side(*, is_ld: bool) -> float:
    """Return the regular cell hexagon side in mm."""

    nc = density_number(is_ld=is_ld)
    return DEFAULT_WAFER_SIDE_MM / (nc * sqrt(3.0))


def valid_cell_coordinates(*, is_ld: bool) -> set[tuple[int, int]]:
    return default_cell_set(hd=not is_ld)


def _raw_local_center(iu: int, iv: int, side: float) -> Point:
    q = iu
    r = iv - iu
    return Point(1.5 * side * q, sqrt(3.0) * side * (r + 0.5 * q))


def local_cells(*, is_ld: bool) -> list[LocalSiliconCell]:
    side = regular_cell_side(is_ld=is_ld)
    coords = sorted(valid_cell_coordinates(is_ld=is_ld))
    raw = [(iu, iv, _raw_local_center(iu, iv, side)) for iu, iv in coords]
    mean_x = sum(point.x for _, _, point in raw) / len(raw)
    mean_y = sum(point.y for _, _, point in raw) / len(raw)
    return [
        LocalSiliconCell(iu=iu, iv=iv, center=Point(point.x - mean_x, point.y - mean_y), side=side)
        for iu, iv, point in raw
    ]


def rotate_point(point: Point, angle_rad: float) -> Point:
    c = cos(angle_rad)
    s = sin(angle_rad)
    return Point(c * point.x - s * point.y, s * point.x + c * point.y)


def _from_hardware(point: Point) -> Point:
    # HXGActiveWafer::fromHardware(), i.e. rotate 150 degrees.
    return Point(-point.x * 0.5 * sqrt(3.0) - point.y * 0.5, point.x * 0.5 - point.y * 0.5 * sqrt(3.0))


def _scale_point(point: Point, scale: float) -> Point:
    return Point(point.x * scale, point.y * scale)


def _dicing_points(*, is_ld: bool, scale: float) -> list[list[Point]]:
    if is_ld:
        z_b = Point(0.8985, 82.3869)
        z_d = Point(47.4375, 77.5032)
        z_e = Point(49.2345, 76.4657)
        z_h = Point(90.8385, 0.8985)
        hardware = [
            [Point(-z_h.x, z_h.y), Point(-z_h.x, -z_h.y), Point(z_h.x, -z_h.y), Point(z_h.x, z_h.y)],
            [Point(-z_b.x, -z_b.y), Point(z_b.x, -z_b.y), Point(z_b.x, z_b.y), Point(-z_b.x, z_b.y)],
            [Point(z_d.x, -z_d.y), Point(z_e.x, -z_e.y), Point(z_e.x, z_e.y), Point(z_d.x, z_d.y)],
        ]
    else:
        z_a = Point(27.2975, 82.3869)
        z_b = Point(29.0945, 82.3869)
        z_e = Point(85.2148, 17.1775)
        z_f = Point(86.2523, 15.3805)
        hardware = [
            [Point(-z_e.x, z_e.y), Point(-z_f.x, z_f.y), Point(z_f.x, z_f.y), Point(z_e.x, z_e.y)],
            [Point(-z_b.x, -z_b.y), Point(-z_a.x, -z_a.y), Point(-z_a.x, z_a.y), Point(-z_b.x, z_b.y)],
            [Point(z_a.x, -z_a.y), Point(z_b.x, -z_b.y), Point(z_b.x, z_b.y), Point(z_a.x, z_a.y)],
        ]
    return [[_scale_point(_from_hardware(point), scale) for point in line] for line in hardware]


def _reference_wafer_corners(side: float) -> list[Point]:
    h = sqrt(0.75) * side
    return [
        Point(0.0, -side),
        Point(h, -0.5 * side),
        Point(h, 0.5 * side),
        Point(0.0, side),
        Point(-h, 0.5 * side),
        Point(-h, -0.5 * side),
    ]


def _transform_local_polygon(points: list[Point], wafer: Wafer) -> list[Point]:
    angle = (wafer.placement % 6) * pi / 3.0
    do_reflect = wafer.placement > 5 or wafer.seen_from_back
    transformed: list[Point] = []
    for point in points:
        # ObjC order: rotate by placement first, then mirror if seen_from_back.
        # M·R(φ) ≠ R(φ)·M, so the order matters for non-zero placement.
        rotated = rotate_point(point, angle)
        if do_reflect:
            rotated = Point(-rotated.x, rotated.y)
        transformed.append(Point(wafer.center.x + rotated.x, wafer.center.y + rotated.y))
    return transformed


def partial_wafer_polygon(wafer: Wafer) -> list[Point]:
    """Return the drawable active polygon for a full or partial wafer.

    The point ordering follows HXGCellView::setUpPartials. The dicing-line
    points come from HXGActiveWafer and are first converted from hardware to the
    local drawing frame, then rotated/mirrored by the wafer placement index.
    Full wafers still return the regular wafer boundary, so edge cells are
    clipped at the wafer boundary instead of being drawn as full hexagons.
    """

    corners = _reference_wafer_corners(wafer.side)
    if wafer.partial_type == 0:
        return _ensure_ccw(_transform_local_polygon(corners, wafer))

    p = wafer.partial_type
    scale = wafer.side / DEFAULT_WAFER_SIDE_MM
    dice = _dicing_points(is_ld=wafer.is_ld, scale=scale)

    if wafer.is_ld:
        if p == 1:
            polygon = [corners[1], corners[4], corners[5], corners[0]]
        elif p == 2:
            polygon = [corners[1], corners[2], corners[3], corners[4]]
        elif p == 3:
            polygon = [dice[1][2], dice[1][1], corners[2], corners[1], corners[0]]
        elif p == 4:
            polygon = [dice[1][0], dice[1][3], corners[5], corners[4], corners[3]]
        elif p == 5:
            polygon = [dice[2][1], dice[2][2], corners[5], corners[0], corners[1], corners[2], corners[3]]
        else:
            polygon = corners
    else:
        if p == 1:
            polygon = [dice[0][2], corners[5], corners[0], dice[0][1]]
        elif p == 2:
            polygon = [dice[0][3], corners[4], corners[3], corners[2], corners[1], dice[0][0]]
        elif p == 3:
            polygon = [dice[1][2], corners[0], corners[1], corners[2], dice[1][1]]
        elif p == 4:
            polygon = [dice[2][0], corners[3], corners[4], corners[5], dice[2][3]]
        else:
            polygon = corners

    return _ensure_ccw(_transform_local_polygon(polygon, wafer))


def transform_to_wafer(cell: LocalSiliconCell, wafer: Wafer) -> SiliconCell | None:
    # Compute the cell center: rotate by placement, then mirror if seen_from_back.
    # This matches HXGWafer::constructWaferBezierMirrored — rotate first, mirror after.
    angle = (wafer.placement % 6) * pi / 3.0
    do_reflect = wafer.placement > 5 or wafer.seen_from_back
    rotated_center = rotate_point(cell.center, angle)
    if do_reflect:
        rotated_center = Point(-rotated_center.x, rotated_center.y)
    center = Point(wafer.center.x + rotated_center.x, wafer.center.y + rotated_center.y)

    # Build the cell polygon using the same rotate-then-reflect pipeline so that
    # the corners are consistent with partial_wafer_polygon for clipping.
    local_corners = [Point(cell.center.x + dx, cell.center.y + dy) for dx, dy in _flat_top_hex_offsets(cell.side)]
    cell_polygon = _transform_local_polygon(local_corners, wafer)

    active_polygon = partial_wafer_polygon(wafer)
    clipped = clip_polygon(cell_polygon, active_polygon)
    if len(clipped) < 3 or polygon_area(clipped) <= MIN_POLYGON_AREA_MM2:
        return None

    return SiliconCell(
        layer=wafer.metadata.get("layer") if isinstance(wafer.metadata.get("layer"), int) else None,
        wafer_u=wafer.u,
        wafer_v=wafer.v,
        iu=cell.iu,
        iv=cell.iv,
        center=center,
        side=cell.side,
        is_ld=wafer.is_ld,
        wafer_type=wafer.partial_type,
        sensor_type=wafer.metadata.get("sensor_type") if isinstance(wafer.metadata.get("sensor_type"), str) else None,
        orientation_rad=angle,
        vertices=tuple(clipped),
    )


def cells_for_wafer(wafer: Wafer) -> list[SiliconCell]:
    """Return regular or clipped grid cells for one wafer."""

    out: list[SiliconCell] = []
    for local_cell in local_cells(is_ld=wafer.is_ld):
        cell = transform_to_wafer(local_cell, wafer)
        if cell is not None:
            out.append(cell)
    return out


def cells_for_wafers(wafers: list[Wafer]) -> list[SiliconCell]:
    out: list[SiliconCell] = []
    for wafer in wafers:
        out.extend(cells_for_wafer(wafer))
    return out
