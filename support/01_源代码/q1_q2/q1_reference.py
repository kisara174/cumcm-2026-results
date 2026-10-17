"""Q1: intersect forward bearing wedges, compute diameter and disk coverage.

Only angular constraints define the requested polygon. No artificial bounding
box, unknown receiver radius, or sampled approximation of the target disk is used.
Floating point results are numerical evidence, not interval certificates.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


def cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def bearing_halfplanes(positions, bearings_deg, error_deg):
    """Return unit-normal A,b for A @ x <= b, with two rows per wedge."""
    positions = np.asarray(positions, dtype=float)
    bearings = np.asarray(bearings_deg, dtype=float)
    if (positions.ndim != 2 or positions.shape[1] != 2
            or len(positions) == 0 or bearings.shape != (len(positions),)
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(bearings))):
        raise ValueError("Require finite positions (n,2) and bearings (n,), n >= 1")
    if not np.isfinite(error_deg) or not 0 < error_deg < 90:
        raise ValueError("error_deg must lie strictly between 0 and 90")
    angles = np.deg2rad(bearings % 360)
    lower, upper = angles - np.deg2rad(error_deg), angles + np.deg2rad(error_deg)
    # cross(lower_ray, x-S) >= 0; cross(upper_ray, x-S) <= 0.
    a = np.empty((2 * len(positions), 2))
    a[0::2] = np.column_stack((np.sin(lower), -np.cos(lower)))
    a[1::2] = np.column_stack((-np.sin(upper), np.cos(upper)))
    b = np.einsum("ij,ij->i", a, np.repeat(positions, 2, axis=0))
    return a, b


def convex_hull(points, tolerance):
    """Monotone-chain hull; return one/two vertices for a point/segment."""
    unique = [np.asarray(p) for p in sorted(set(map(tuple, points)))]
    if len(unique) <= 2:
        return np.asarray(unique)

    def chain(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and cross(out[-1] - out[-2], p - out[-1]) <= 0:
                out.pop()
            out.append(p)
        return out

    hull = chain(unique)[:-1] + chain(unique[::-1])[:-1]
    cleaned = []
    for p in hull:
        if not cleaned or np.linalg.norm(p - cleaned[-1]) > tolerance:
            cleaned.append(p)
    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= tolerance:
        cleaned.pop()
    return np.asarray(cleaned)


def solve(positions, bearings_deg, error_deg, tolerance=1e-9, parallel_tol=1e-12):
    """Enumerate feasible boundary intersections, then all hull vertex pairs.

    Empty and unbounded angular intersections are reported explicitly. LPs use
    unrestricted x/y, so negative coordinates are permitted. A translation and
    scale improve conditioning; `tolerance` is measured in these scaled units.
    """
    if not 1e-10 <= tolerance <= 1e-4 or not 0 < parallel_tol < 1:
        raise ValueError("Invalid numerical tolerances")
    positions = np.asarray(positions, dtype=float)
    # Validate before centering (in particular, before a mean of an empty array).
    bearing_halfplanes(positions, bearings_deg, error_deg)
    origin = positions.mean(axis=0)
    scale = max(1.0, float(np.max(np.linalg.norm(positions - origin, axis=1))))
    local_positions = (positions - origin) / scale
    a, b = bearing_halfplanes(local_positions, bearings_deg, error_deg)
    options = {"primal_feasibility_tolerance": tolerance,
               "dual_feasibility_tolerance": tolerance}

    def lp(objective):
        result = linprog(objective, A_ub=a, b_ub=b, bounds=[(None, None)] * 2,
                         method="highs", options=options)
        if result.status not in (0, 2, 3):
            raise ArithmeticError(f"LP failed: {result.message}")
        return result

    result = {
        "geometry": "intersection_of_forward_angular_wedges",
        "bearing_error_deg": float(error_deg),
        "length_tolerance_m": float(tolerance * scale),
        "formal_interval_certificate": False,
    }
    feasibility = lp([0, 0])
    if feasibility.status == 2:
        return dict(result, status="empty", diameter_m=None,
                    same_diameter_disk_covers=None, vertices_m=[])
    for direction in ([1, 0], [-1, 0], [0, 1], [0, -1]):
        bound = lp(direction)
        if bound.status == 3:
            return dict(result, status="unbounded", diameter_m=None,
                        diameter_is_infinite=True,
                        same_diameter_disk_covers=None, vertices_m=[])
        if bound.status != 0:
            raise ArithmeticError("Inconsistent LP feasibility decisions")

    candidates = []
    skipped_pairs = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        matrix = a[[i, j]]
        if abs(cross(matrix[0], matrix[1])) <= parallel_tol:
            skipped_pairs += 1
            continue
        point = np.linalg.solve(matrix, b[[i, j]])
        if np.max(a @ point - b) <= tolerance:
            candidates.append(point)
    if not candidates:
        raise ArithmeticError("Bounded LP but no resolved vertices; inspect conditioning")
    vertices = convex_hull(candidates, tolerance)
    # Independent LP support values guard against silently omitted extreme points.
    max_support_error = 0.0
    for direction in np.asarray([[1, 0], [-1, 0], [0, 1], [0, -1]]):
        support_error = abs(float(np.min(vertices @ direction)) - lp(direction).fun)
        max_support_error = max(max_support_error, support_error)
    if max_support_error > 10 * tolerance:
        raise ArithmeticError("Enumerated vertices disagree with LP support; ill-conditioned input")

    pair = (0, 0)
    diameter_squared = 0.0
    for i, j in itertools.combinations(range(len(vertices)), 2):
        squared = float(np.sum((vertices[i] - vertices[j]) ** 2))
        if squared > diameter_squared:
            diameter_squared, pair = squared, (i, j)
    diameter = float(np.sqrt(diameter_squared))
    midpoint = (vertices[pair[0]] + vertices[pair[1]]) / 2
    distances = np.linalg.norm(vertices - midpoint, axis=1)
    farthest = int(np.argmax(distances))
    excess = float(distances[farthest] - diameter / 2)
    world_vertices = vertices * scale + origin
    return dict(
        result,
        status={1: "point", 2: "segment"}.get(len(vertices), "polygon"),
        vertices_m=world_vertices.tolist(),
        diameter_m=diameter * scale,
        diameter_endpoint_indices=list(pair),
        diameter_endpoints_m=world_vertices[list(pair)].tolist(),
        diameter_disk_center_m=(midpoint * scale + origin).tolist(),
        diameter_disk_radius_m=diameter * scale / 2,
        same_diameter_disk_covers=bool(excess <= tolerance),
        coverage_decision="within_numerical_tolerance" if excess <= tolerance else "does_not_cover",
        maximum_distance_from_diameter_midpoint_m=float(max(distances) * scale),
        maximum_radius_excess_m=excess * scale,
        farthest_vertex_index=farthest,
        max_halfplane_residual_m=float(np.max(a @ vertices.T - b[:, None]) * scale),
        max_lp_support_error_m=max_support_error * scale,
        parallel_or_near_parallel_pairs_skipped=skipped_pairs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON with positions_m and bearings_deg")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs/q1.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = solve(data["positions_m"], data["bearings_deg"],
                   config["bearing_error_deg"],
                   config["normalized_feasibility_tolerance"],
                   config["parallel_determinant_tolerance"])
    result["provenance"] = {
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "solver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_description": data.get("description", "user-supplied bearings"),
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
