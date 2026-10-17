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
from scipy.spatial import HalfspaceIntersection, QhullError

from q1_geometry import minimum_enclosing_disk, polygon_diameter


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


def solve(positions, bearings_deg, error_deg, tolerance=1e-9, parallel_tol=1e-12,
          *, intersection_method="auto", clearance_radius_m=20.0, mec_seed=20260910):
    """Qhull intersection with audited enumeration fallback; calipers and MEC.

    Empty and unbounded angular intersections are reported explicitly. LPs use
    unrestricted x/y, so negative coordinates are permitted. A translation and
    scale improve conditioning; `tolerance` is measured in these scaled units.
    """
    if not 1e-10 <= tolerance <= 1e-4 or not 0 < parallel_tol < 1:
        raise ValueError("Invalid numerical tolerances")
    if intersection_method not in ("auto", "enumeration"):
        raise ValueError("intersection_method must be auto or enumeration")
    if not np.isfinite(clearance_radius_m) or clearance_radius_m < 0:
        raise ValueError("clearance_radius_m must be finite and nonnegative")
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
        "intersection_method_requested": intersection_method,
        "clearance_radius_m": float(clearance_radius_m),
        "minimum_enclosing_radius_m": None,
        "minimum_enclosing_center_m": None,
        "clearance_disk_covers": None,
        "clearance_decision": "not_applicable",
    }
    feasibility = lp([0, 0])
    if feasibility.status == 2:
        return dict(result, status="empty", diameter_m=None,
                    same_diameter_disk_covers=None, vertices_m=[])
    directions = np.asarray([[1, 0], [-1, 0], [0, 1], [0, -1]])
    support_values = []
    for direction in directions:
        bound = lp(direction)
        if bound.status == 3:
            return dict(result, status="unbounded", diameter_m=None,
                        diameter_is_infinite=True,
                        same_diameter_disk_covers=None, vertices_m=[])
        if bound.status != 0:
            raise ArithmeticError("Inconsistent LP feasibility decisions")
        support_values.append(bound.fun)

    def audit(vertices):
        if len(vertices) == 0 or not np.all(np.isfinite(vertices)):
            raise ArithmeticError("Unresolved intersection vertices")
        residual = float(np.max(a @ vertices.T - b[:, None]))
        support_error = float(np.max(np.abs(np.min(vertices @ directions.T, axis=0)
                                            - support_values)))
        if residual > 10 * tolerance or support_error > 10 * tolerance:
            raise ArithmeticError("Vertices disagree with halfplanes or LP support")
        return residual, support_error

    vertices, fallback_reason = None, None
    skipped_pairs = 0
    if intersection_method == "auto":
        # Unit normals: maximize the radius of an inscribed disk to find a
        # strictly interior point. No source location or artificial box is used.
        interior = linprog([0, 0, -1], A_ub=np.column_stack((a, np.ones(len(a)))),
                           b_ub=b, bounds=[(None, None)] * 2 + [(0, None)],
                           method="highs", options=options)
        if interior.status == 0 and interior.x[2] > 32 * tolerance:
            try:
                points = HalfspaceIntersection(np.column_stack((a, -b)),
                                               interior.x[:2]).intersections
                vertices = convex_hull(points, tolerance)
                audit(vertices)
            except (QhullError, ArithmeticError, FloatingPointError) as exc:
                vertices = None
                fallback_reason = type(exc).__name__ + ": fast intersection rejected"
        else:
            fallback_reason = "no_resolved_strict_interior"
    if vertices is None:
        candidates = []
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
        used_method = "enumeration"
    else:
        used_method = "qhull"
    residual, max_support_error = audit(vertices)
    diameter, pair = polygon_diameter(vertices)
    midpoint = (vertices[pair[0]] + vertices[pair[1]]) / 2
    distances = np.linalg.norm(vertices - midpoint, axis=1)
    farthest = int(np.argmax(distances))
    excess = float(distances[farthest] - diameter / 2)
    world_vertices = vertices * scale + origin
    disk = minimum_enclosing_disk(vertices, tolerance=tolerance, seed=mec_seed)
    radius, lower = disk["radius"] * scale, disk["lower_bound"] * scale
    length_tolerance = tolerance * scale
    if radius < clearance_radius_m - length_tolerance:
        clearance_decision = "covers_with_margin"
        clearance_covers = True
    elif lower > clearance_radius_m + length_tolerance:
        clearance_decision = "does_not_cover"
        clearance_covers = False
    else:
        clearance_decision = "boundary_requires_review"
        clearance_covers = None
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
        max_halfplane_residual_m=residual * scale,
        max_lp_support_error_m=max_support_error * scale,
        parallel_or_near_parallel_pairs_skipped=skipped_pairs,
        intersection_method_used=used_method,
        intersection_fallback_reason=fallback_reason,
        diameter_method="rotating_calipers",
        minimum_enclosing_method="welzl_with_support_audit",
        minimum_enclosing_seed=int(mec_seed),
        minimum_enclosing_center_m=(disk["center"] * scale + origin).tolist(),
        minimum_enclosing_radius_m=radius,
        minimum_enclosing_radius_lower_bound_m=lower,
        minimum_enclosing_optimality_gap_m=disk["optimality_gap"] * scale,
        minimum_enclosing_support_indices=disk["support_indices"],
        minimum_enclosing_support_weights=disk["support_weights"],
        clearance_disk_covers=clearance_covers,
        clearance_decision=clearance_decision,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON with positions_m and bearings_deg")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs/q1.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", choices=("auto", "enumeration"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = solve(data["positions_m"], data["bearings_deg"],
                   config["bearing_error_deg"],
                   config["normalized_feasibility_tolerance"],
                   config["parallel_determinant_tolerance"],
                   intersection_method=args.method or config.get("intersection_method", "auto"),
                   clearance_radius_m=config.get("clearance_radius_m", 20.0),
                   mec_seed=config.get("minimum_enclosing_seed", 20260910))
    def text_hash(path):
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()

    result["provenance"] = {
        "hash_scheme": "sha256_text_lf",
        "input_sha256": text_hash(args.input),
        "config_sha256": text_hash(args.config),
        "solver_sha256": text_hash(Path(__file__)),
        "geometry_sha256": text_hash(Path(__file__).with_name("q1_geometry.py")),
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
