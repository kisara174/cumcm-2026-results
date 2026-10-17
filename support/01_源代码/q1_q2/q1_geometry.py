"""Geometry primitives for Q1; floating-point audits, not interval proofs."""
import itertools
import numpy as np


def cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def polygon_diameter(vertices):
    """O(k) rotating calipers on a CCW convex hull, including parallel edges."""
    v = np.asarray(vertices, dtype=float)
    n = len(v)
    if n == 0:
        raise ValueError("Diameter of empty polygon is undefined")
    if n <= 2:
        return float(np.linalg.norm(v[-1] - v[0])), (0, n - 1)
    best, pair, j = -1.0, (0, 0), 1
    for i in range(n):
        ni = (i + 1) % n
        edge = v[ni] - v[i]
        steps = 0
        while cross(edge, v[(j + 1) % n] - v[i]) > cross(edge, v[j] - v[i]):
            j = (j + 1) % n
            steps += 1
            if steps > n:
                raise ArithmeticError("Invalid convex hull in rotating calipers")
        # Testing the next endpoint also covers tied parallel support edges.
        for a, b in ((i, j), (ni, j), (i, (j + 1) % n), (ni, (j + 1) % n)):
            delta = v[a] - v[b]
            squared = float(delta @ delta)
            candidate = tuple(sorted((a, b)))
            if squared > best or (squared == best and candidate < pair):
                best, pair = squared, candidate
    return float(np.sqrt(best)), pair


def minimum_enclosing_disk(points, tolerance=1e-9, seed=20260910):
    """Welzl with an explicit stack (no recursion limit), then a dual audit.

    Random permutation gives expected O(k) arithmetic work, not worst-case O(k).
    A fixed seed makes the run reproducible. The returned upper radius covers
    all input points. A nonnegative weighted support variance is a lower bound
    on every enclosing radius; a large upper/lower gap raises ArithmeticError.
    """
    original = np.asarray(points, dtype=float)
    if (original.ndim != 2 or original.shape[1] != 2 or not len(original)
            or not np.all(np.isfinite(original)) or not np.isfinite(tolerance)
            or tolerance <= 0):
        raise ValueError("Require finite nonempty (k,2) points and positive tolerance")
    # Additional centering avoids cancellation when called outside solve().
    origin = original[0]
    scale = max(1.0, float(np.linalg.norm(original - origin, axis=1).max()))
    normalized = (original - origin) / scale
    order = np.random.default_rng(seed).permutation(len(original))
    v = normalized[order]
    eps = 64 * np.finfo(float).eps

    def contains(disk, p):
        if disk is None:
            return False
        c, r, _ = disk
        return np.linalg.norm(p - c) <= r + eps * max(1.0, r)

    def boundary_disk(indices):
        if not indices:
            return None
        if len(indices) == 1:
            return v[indices[0]].copy(), 0.0, indices
        candidates = []
        for i, j in itertools.combinations(indices, 2):
            center = (v[i] + v[j]) / 2
            disk = (center, float(np.linalg.norm(v[i] - center)), (i, j))
            if all(contains(disk, v[t]) for t in indices):
                candidates.append(disk)
        if candidates:
            return min(candidates, key=lambda d: d[1])
        i, j, k = indices
        a, b = v[j] - v[i], v[k] - v[i]
        matrix = 2 * np.asarray([a, b])
        try:
            center = v[i] + np.linalg.solve(matrix, [a @ a, b @ b])
        except np.linalg.LinAlgError as exc:
            raise ArithmeticError("Unresolved collinear disk support") from exc
        return center, float(np.linalg.norm(center - v[i])), indices

    # Frames: prefix length, forced boundary indices, return phase.
    stack = [(len(v), (), 0)]
    disk = None
    while stack:
        n, boundary, phase = stack.pop()
        if phase == 0:
            if n == 0 or len(boundary) == 3:
                disk = boundary_disk(boundary)
            else:
                stack.append((n, boundary, 1))
                stack.append((n - 1, boundary, 0))
        elif not contains(disk, v[n - 1]):
            stack.append((n - 1, boundary + (n - 1,), 0))

    center, _, support = disk
    support_indices = order[list(support)]
    s = normalized[support_indices]
    # Compute weights in a local, scaled coordinate system, not world coordinates.
    if len(s) == 1:
        weights = np.ones(1)
    elif len(s) == 2:
        weights = np.array([0.5, 0.5])
    else:
        local_scale = max(float(np.linalg.norm(s - center, axis=1).max()), 1e-300)
        weights = np.linalg.solve(np.vstack(((s - center).T / local_scale,
                                             np.ones(3))), [0, 0, 1])
    if not np.all(np.isfinite(weights)) or np.min(weights) < -1e-8:
        raise ArithmeticError("Disk center lies outside its support hull")
    weights = np.maximum(weights, 0)
    weights /= weights.sum()
    # For any c, max ||s_i-c||² >= weighted variance of the support.
    mean = weights @ s
    lower = float(np.sqrt(np.sum(weights * np.sum((s - mean) ** 2, axis=1))))
    upper = float(np.linalg.norm(normalized - center, axis=1).max())
    gap = max(0.0, upper - lower) * scale
    if gap > 10 * tolerance:
        raise ArithmeticError("Enclosing disk failed upper/lower radius audit")
    return {
        "center": center * scale + origin,
        "radius": upper * scale,
        "lower_bound": lower * scale,
        "optimality_gap": gap,
        "support_indices": support_indices.tolist(),
        "support_weights": weights.tolist(),
    }
