"""Conservative polar enclosure of the target disk and analytic reception tests.

Cells cover whole continuous angular intervals, not sampled source points.
Float calculations with explicit margins are not outward-rounded certificates.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.optimize import minimize_scalar


def angular_dot_extrema(point, lower, upper):
    """Exact extrema of point dot (cos(alpha),sin(alpha)) on short arcs."""
    a, b = point
    lower, upper = np.asarray(lower), np.asarray(upper)
    ends = np.stack((a*np.cos(lower)+b*np.sin(lower),
                     a*np.cos(upper)+b*np.sin(upper)))
    lo, hi = ends.min(axis=0), ends.max(axis=0)
    phi, radius = np.arctan2(b, a), np.hypot(a, b)
    for shift in (-2*np.pi, 0., 2*np.pi):
        p = phi+shift
        hi = np.where((lower <= p) & (p <= upper), radius, hi)
        p += np.pi
        lo = np.where((lower <= p) & (p <= upper), -radius, lo)
    return lo, hi


def prior_cells(first, bearing_deg, error_deg, config):
    """Cover the actual disk/wedge/range intersection by closed annular sectors.

    At each angle h=c dot u, ray/disk roots are h +/- sqrt(h*h-C).
    The outer root increases with h; when S1 is outside the disk the inner
    root decreases with h. Thus ONE analytic maximum of h per angular cell
    gives outward radial bounds for every ray in that cell, including tangent
    rays and cells that partly miss the disk.
    """
    theta = np.deg2rad(bearing_deg % 360)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    center = -rotation.T @ np.asarray(first, dtype=float)
    delta = np.deg2rad(error_deg)
    cuts = np.linspace(-delta, delta, config['prior_angular_cells']+1)
    _, hmax = angular_dot_extrema(center, cuts[:-1], cuts[1:])
    H = config['target_region_radius_m']
    C = float(np.dot(center, center)-H*H)
    maximum, near = config['maximum_reception_radius_m'], config['near_radius_m']
    padding = config['geometry_margin_m']
    cells = []
    for lo, hi, h in zip(cuts[:-1], cuts[1:], hmax):
        if C > 0 and (h <= 0 or h*h < C):
            continue
        root = np.sqrt(max(0., h*h-C))
        outer = h+root if h >= 0 else -C/(root-h)
        inner = C/(h+root) if C > 0 else 0.
        rlo, rhi = max(near, inner), min(maximum, outer)
        if rhi <= near or rlo > rhi:
            continue
        cells.append([float(lo), float(hi), max(0., float(rlo-padding)), float(rhi+padding)])
    if not cells:
        raise ValueError('First direction is inconsistent with the target disk and reception range')
    return np.asarray(cells), rotation


def reflect_cells(cells, side):
    if side == 1:
        return cells
    result = cells.copy()
    result[:, 0], result[:, 1] = -cells[:, 1], -cells[:, 0]
    return result


def reception_margins(point, cells, minimum_radius):
    """Worst reception slack and minimum target distance over entire cells.

    For r<=R0, squared distance is convex in r. For r>=R0,
    squared-distance minus r*r is affine in r. Only endpoints and R0
    need checking, combined with exact angular dot extrema.
    """
    lower, upper, rlo, rhi = cells.T
    dotmin, dotmax = angular_dot_extrema(point, lower, upper)
    radii = np.stack((rlo, rhi, np.clip(minimum_radius, rlo, rhi)), axis=1)
    rho2 = float(np.dot(point, point))
    distances = np.sqrt(np.maximum(0., rho2+radii*radii-2*radii*dotmin[:, None]))
    slack = float(np.min(np.maximum(minimum_radius, radii)-distances))
    closest_r = np.clip(dotmax, rlo, rhi)
    nearest = float(np.sqrt(np.maximum(0., rho2+closest_r**2-2*closest_r*dotmax)).min())
    return slack, nearest


def reading_interval(point, cells, error_deg):
    """Continuous outer envelope of all second readings, for q above cells.

    Boundary arc extrema occur at endpoints or point dot u = r.
    Straight radial edges have monotone argument. No source sampling is used.
    """
    a, b = point
    lower, upper, rlo, rhi = cells.T
    # A sufficient separating strip keeps the angle branch continuous in (-pi,0).
    max_y = np.maximum(rlo*np.sin(upper), rhi*np.sin(upper)).max()
    if b <= max_y:
        raise ValueError('Second point must be strictly above the prior enclosure')
    radii = np.repeat(np.stack((rlo, rhi), axis=1), 2, axis=1)
    angles = np.tile(np.stack((lower, upper), axis=1), (1, 2))
    values = list(np.arctan2(radii*np.sin(angles)-b, radii*np.cos(angles)-a).ravel())
    phi, rho = np.arctan2(b, a), np.hypot(a, b)
    for rs in (rlo, rhi):
        psi = np.arccos(np.clip(rs/rho, -1, 1))
        for alphas in (phi-psi, phi+psi):
            ok = (rs <= rho) & (lower <= alphas) & (alphas <= upper)
            values.extend(np.arctan2(rs[ok]*np.sin(alphas[ok])-b,
                                     rs[ok]*np.cos(alphas[ok])-a))
    delta = np.deg2rad(error_deg)
    return float(min(values)-delta), float(max(values)+delta)


def diameters(point, theta, error_deg):
    """Batch two-wedge geometry; q is outside first wedge and crossings bounded."""
    theta = np.atleast_1d(theta)
    delta, n = np.deg2rad(error_deg), len(theta)
    A = np.empty((n, 4, 2))
    A[:, 0] = [np.sin(-delta), -np.cos(-delta)]
    A[:, 1] = [-np.sin(delta), np.cos(delta)]
    A[:, 2, 0], A[:, 2, 1] = np.sin(theta-delta), -np.cos(theta-delta)
    A[:, 3, 0], A[:, 3, 1] = -np.sin(theta+delta), np.cos(theta+delta)
    rhs = np.zeros((n, 4))
    rhs[:, 2:] = A[:, 2:] @ np.asarray(point)
    vertices = np.empty((n, 6, 2))
    valid = np.empty((n, 6), dtype=bool)
    tol = 1e-11*max(1., np.linalg.norm(point))
    for k, (i, j) in enumerate(itertools.combinations(range(4), 2)):
        det = A[:, i, 0]*A[:, j, 1]-A[:, i, 1]*A[:, j, 0]
        if np.any(np.abs(det) < 1e-12):
            raise ArithmeticError('Unresolved parallel boundary')
        vertices[:, k, 0] = (rhs[:, i]*A[:, j, 1]-A[:, i, 1]*rhs[:, j])/det
        vertices[:, k, 1] = (A[:, i, 0]*rhs[:, j]-rhs[:, i]*A[:, j, 0])/det
        valid[:, k] = np.max(np.einsum('nij,nj->ni', A, vertices[:, k])-rhs, axis=1) <= tol
    if np.any(valid.sum(axis=1) < 1):
        raise ArithmeticError('Unresolved two-wedge intersection')
    result = np.zeros(n)
    for i, j in itertools.combinations(range(6), 2):
        ds = np.sum((vertices[:, i]-vertices[:, j])**2, axis=1)
        result = np.maximum(result, np.where(valid[:, i] & valid[:, j], ds, 0.))
    return np.sqrt(result)


def evaluate(point, interval, error_deg, samples=129, refine=False):
    low, high = interval
    delta = np.deg2rad(error_deg)
    origin = np.arctan2(-point[1], -point[0])
    # Include exact topology changes, even during training: a uniform grid can
    # miss a cusp where the second wedge starts/stops containing the origin.
    angles = np.unique(np.r_[np.linspace(low, high, samples),
                            [t for t in (origin-delta, origin+delta) if low < t < high]])
    ds = diameters(point, angles, error_deg)
    k = int(ds.argmax())
    best, reading = float(ds[k]), float(angles[k])
    if refine:
        peaks = np.flatnonzero((ds[1:-1] >= ds[:-2]) & (ds[1:-1] >= ds[2:]))+1
        for i in peaks:
            opt = minimize_scalar(lambda t: -diameters(point, [t], error_deg)[0],
                                  bounds=(angles[i-1], angles[i+1]), method='bounded',
                                  options={'xatol': 1e-12})
            if opt.success and -opt.fun > best:
                best, reading = float(-opt.fun), float(opt.x)
    return {'sampled_worst_diameter_m': best, 'worst_second_bearing_relative_deg': float(np.rad2deg(reading)),
            'second_bearing_interval_relative_deg': np.rad2deg(interval).tolist(),
            'bearing_samples': int(samples), 'topology_changes_included': True,
            'local_peak_refinement': bool(refine), 'formal_continuous_certificate': False}
