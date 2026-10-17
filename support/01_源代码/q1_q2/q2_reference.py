"""Q2: reception-safe second detector selection with sampled minimax diameter.

Analytical reception guarantee, explicit numerical search (no global-optimality
claim). Coordinates local to the first detector and its observed bearing.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

import q1


def safe_disk_centers(error_deg, minimum_radius):
    delta = np.deg2rad(error_deg)
    return np.array([[0., 0.], [minimum_radius * np.cos(delta), minimum_radius * np.sin(delta)],
                     [minimum_radius * np.cos(delta), -minimum_radius * np.sin(delta)]])


def reception_safe(point, error_deg, minimum_radius, tolerance=1e-8):
    """Exact three-disk sufficient region, with numerical boundary tolerance."""
    point = np.asarray(point, dtype=float)
    return bool(np.all(np.linalg.norm(safe_disk_centers(error_deg, minimum_radius) - point,
                                     axis=1) <= minimum_radius + tolerance))


def bearing_interval(point, error_deg, maximum_radius):
    """All possible second readings for the enlarged first-bearing sector.

    Use the upper side b > L*sin(delta); the lower side is its reflection.
    The sector includes r=0 as a conservative closure, although direction
    readings physically require r>5. Return continuous radians in (-pi,0).
    """
    a, b = point
    delta = np.deg2rad(error_deg)
    if not (0 < a < maximum_radius * np.cos(delta)
            and b > maximum_radius * np.sin(delta)
            and np.hypot(a, b) < maximum_radius):
        raise ValueError("Point must be above the full prior sector and inside radius L")
    return (float(np.arctan2(-b, -a) - delta),
            float(np.arctan2(maximum_radius * np.sin(delta) - b,
                             maximum_radius * np.cos(delta) - a) + delta))


def candidate(point, error_deg, config):
    """Upper branch: safe reception plus measured acute crossing angle >= gamma."""
    point = np.asarray(point, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        return False
    if not reception_safe(point, error_deg, config["minimum_reception_radius_m"]):
        return False
    try:
        low, high = bearing_interval(point, error_deg, config["maximum_reception_radius_m"])
    except ValueError:
        return False
    gamma = np.deg2rad(config["minimum_measured_crossing_angle_deg"])
    return bool(low >= -np.pi + gamma and high <= -gamma)


def two_wedge_diameters(point, second_bearings_rad, error_deg):
    """Vectorized exact line-intersection geometry for a batch of second bearings.

    Caller uses the candidate region: nonempty bounded two-wedge intersections,
    and every boundary pair across different detectors is nonparallel. This is
    the same polygon metric as Q1; no target disk is approximated or clipped.
    """
    theta = np.atleast_1d(np.asarray(second_bearings_rad, dtype=float))
    delta = np.deg2rad(error_deg)
    n = len(theta)
    a = np.empty((n, 4, 2))
    a[:, 0] = [np.sin(-delta), -np.cos(-delta)]
    a[:, 1] = [-np.sin(delta), np.cos(delta)]
    a[:, 2, 0], a[:, 2, 1] = np.sin(theta - delta), -np.cos(theta - delta)
    a[:, 3, 0], a[:, 3, 1] = -np.sin(theta + delta), np.cos(theta + delta)
    b = np.zeros((n, 4))
    b[:, 2:] = a[:, 2:] @ np.asarray(point)
    vertices = np.empty((n, 6, 2))
    valid = np.empty((n, 6), dtype=bool)
    scale = max(1., float(np.linalg.norm(point)))
    for k, (i, j) in enumerate(itertools.combinations(range(4), 2)):
        det = a[:, i, 0] * a[:, j, 1] - a[:, i, 1] * a[:, j, 0]
        if np.any(np.abs(det) < 1e-12):
            raise ArithmeticError("Parallel boundary outside supported candidate geometry")
        vertices[:, k, 0] = (b[:, i] * a[:, j, 1] - a[:, i, 1] * b[:, j]) / det
        vertices[:, k, 1] = (a[:, i, 0] * b[:, j] - b[:, i] * a[:, j, 0]) / det
        residual = np.einsum("nij,nj->ni", a, vertices[:, k]) - b
        valid[:, k] = np.max(residual, axis=1) <= 1e-9 * scale
    if np.any(np.sum(valid, axis=1) < 1):
        raise ArithmeticError("No feasible vertices; unsupported or ill-conditioned geometry")
    diameters_squared = np.zeros(n)
    for i, j in itertools.combinations(range(6), 2):
        distances = np.sum((vertices[:, i] - vertices[:, j]) ** 2, axis=1)
        diameters_squared = np.maximum(diameters_squared,
                                       np.where(valid[:, i] & valid[:, j], distances, 0.))
    return np.sqrt(diameters_squared)


def evaluate(point, error_deg, maximum_radius, samples, refine=False):
    low, high = bearing_interval(point, error_deg, maximum_radius)
    bearings = np.linspace(low, high, samples)
    diameters = two_wedge_diameters(point, bearings, error_deg)
    index = int(np.argmax(diameters))
    best_theta, best_diameter = float(bearings[index]), float(diameters[index])
    if refine:
        # Refine sampled local maxima; this is a numerical adversarial check,
        # not a proof that no narrower unsampled peak exists.
        peaks = np.flatnonzero((diameters[1:-1] >= diameters[:-2])
                              & (diameters[1:-1] >= diameters[2:])) + 1
        for i in peaks:
            opt = minimize_scalar(lambda t: -two_wedge_diameters(point, [t], error_deg)[0],
                                  bounds=(bearings[i - 1], bearings[i + 1]),
                                  method="bounded", options={"xatol": 1e-12})
            if opt.success and -opt.fun > best_diameter:
                best_theta, best_diameter = float(opt.x), float(-opt.fun)
    return {
        "second_bearing_interval_relative_deg": np.rad2deg([low, high]).tolist(),
        "sampled_worst_diameter_m": best_diameter,
        "worst_second_bearing_relative_deg": float(np.rad2deg(best_theta)),
        "bearing_samples": samples,
        "local_peak_refinement": refine,
    }


def choose(first_position, first_bearing_deg, geometry_config, config):
    first = np.asarray(first_position, dtype=float)
    if first.shape != (2,) or not np.all(np.isfinite(first)) or not np.isfinite(first_bearing_deg):
        raise ValueError("Require finite first_position_m[2] and first_bearing_deg")
    error = float(geometry_config["bearing_error_deg"])
    radius, maximum = config["minimum_reception_radius_m"], config["maximum_reception_radius_m"]
    gamma = config["minimum_measured_crossing_angle_deg"]
    if not (0 < error < 45 and 2 * error < gamma < 90
            and 0 < radius < maximum and config["walking_speed_m_s"] > 0):
        raise ValueError("Unsupported physical/angle configuration")
    steps = config["candidate_grid_steps_m"]
    if (not steps or any(not np.isfinite(s) or s <= 0 for s in steps)
            or any(b >= a for a, b in zip(steps, steps[1:]))
            or config["training_bearing_samples"] < 3
            or config["validation_bearing_samples"] < 3):
        raise ValueError("Invalid grid settings")
    evaluated = {}
    history = []
    best = None
    for level, step in enumerate(steps):
        if level == 0:
            xs = ys = np.arange(step, radius + 1e-8, step)
        else:
            span = steps[level - 1]
            xs = np.arange(max(step, best[0] - span), min(radius, best[0] + span) + step / 2, step)
            ys = np.arange(max(step, best[1] - span), min(radius, best[1] + span) + step / 2, step)
        for a in xs:
            for b in ys:
                p = (float(a), float(b))
                if p in evaluated or not candidate(p, error, config):
                    continue
                metrics = evaluate(p, error, maximum, config["training_bearing_samples"])
                evaluated[p] = metrics["sampled_worst_diameter_m"]
        if not evaluated:
            raise ValueError("No candidate found; adjust explicit angle/grid settings")
        best = min(evaluated, key=lambda p: (evaluated[p], np.hypot(*p), p))
        history.append({"step_m": step, "cumulative_candidates": len(evaluated),
                        "selected_local_m": list(best), "sampled_worst_diameter_m": evaluated[best]})

    theta = np.deg2rad(first_bearing_deg % 360)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    local = np.array(best)
    selected = first + rotation @ local
    reflected = first + rotation @ (local * [1, -1])
    audit = evaluate(best, error, maximum, config["validation_bearing_samples"], refine=True)
    duration = np.linalg.norm(local) / config["walking_speed_m_s"] + config["measurement_time_s"]
    # A simple forward/lateral candidate used for a like-for-like conservative comparison.
    reference = [0.5 * radius, 0.5 * radius]
    comparison = None
    if candidate(reference, error, config):
        comparison = dict(local_position_m=reference,
                          **evaluate(reference, error, maximum, config["validation_bearing_samples"], True))
    return {
        "status": "selected", "first_position_m": first.tolist(),
        "first_bearing_deg": float(first_bearing_deg % 360),
        "second_position_m": selected.tolist(), "symmetric_second_position_m": reflected.tolist(),
        "selected_local_m": local.tolist(),
        "move_distance_m": float(np.linalg.norm(local)),
        "move_and_measure_time_s": float(duration),
        "same_channel_switch_time_s": 0.,
        "safe_region_local_disk_centers_m": safe_disk_centers(error, radius).tolist(),
        "safe_region_disk_radius_m": radius,
        "minimum_measured_crossing_angle_deg": gamma,
        "safe_region_margin_m": float(radius - np.max(np.linalg.norm(safe_disk_centers(error, radius) - local, axis=1))),
        "selection_metric": "sampled_minimax_of_Q1_angular_polygon_diameter",
        "prior": {"radial_interval_m": [0., maximum], "angular_error_deg": error,
                  "target_disk_used_to_tighten": False, "near_radius_used_to_tighten": False},
        "global_optimality_claimed": False, "formal_interval_certificate": False,
        "grid_history": history, "audit": audit, "reference_candidate": comparison,
        "reception_guarantee_basis": "analytical three-disk region; proof in q2-solution.md",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    configs = Path(__file__).resolve().parents[1] / "configs"
    parser.add_argument("--config", type=Path, default=configs / "q2.json")
    parser.add_argument("--geometry-config", type=Path, default=configs / "q1.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    geometry = json.loads(args.geometry_config.read_text(encoding="utf-8"))
    result = choose(data["first_position_m"], data["first_bearing_deg"], geometry, config)
    result["provenance"] = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in {
        "input_sha256": args.input, "config_sha256": args.config,
        "geometry_config_sha256": args.geometry_config, "solver_sha256": Path(__file__),
        "q1_solver_sha256": Path(q1.__file__),
    }.items()}
    result["input_description"] = data.get("description", "user-supplied first bearing")
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
