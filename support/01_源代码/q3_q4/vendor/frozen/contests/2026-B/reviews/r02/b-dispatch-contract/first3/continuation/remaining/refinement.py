"""Reviewed opt-in integration; archived joint.py and evidence remain unchanged.

Certified query answers are sufficient: the entire exact physical set need not
be solved before a safe action is chosen. No continuous sensing optimum claim.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import heapq
import itertools
import math

import certified as cert
from joint import Controller, action, exact_cost, difference_bounds
from optimization import clear_center
import core


def count_supports(domains, minimum=10, maximum=16):
    """Exact support projection for minimum <= sum(binary existence) <= maximum.

Counts include already-cleared sources. Independent binary domains plus this
one cardinality constraint; not exact inference for all physical constraints.
"""
    domains = tuple(frozenset(d) for d in domains)
    if not 0 <= minimum <= maximum <= len(domains):
        raise ValueError('invalid source-count bounds')
    if any(not d or not d <= {0, 1} for d in domains):
        raise ValueError('invalid or empty existence domain')
    lower, upper = sum(min(d) for d in domains), sum(max(d) for d in domains)
    if lower > maximum or upper < minimum:
        raise ValueError('source count inconsistent with confirmed observations')
    return tuple(frozenset(v for v in d
                          if lower-min(d)+v <= maximum and upper-max(d)+v >= minimum)
                 for d in domains)


def point_box_distance2(p, box):
    return sum(max(lo-cert.rat(x), F(0), cert.rat(x)-hi)**2
               for x, (lo, hi) in zip(p, box))


def route_center(poly, start, following=None, max_splits=128, tolerance_s=F(1, 100)):
    """Constrained continuous travel minimization over guaranteed-clear centers.

min (|start-q| + |q-following|)/5 subject to max_v |q-v| <=20.
The search domain contains ALL feasible centers, not just the source bbox.
Returns a feasible command with valid bounds, or undetermined (never 'empty'
merely because a finite search did not find a witness).
"""
    poly = tuple(tuple(map(cert.rat, v)) for v in poly)
    if not poly or max_splits < 0 or tolerance_s <= 0:
        raise ValueError('invalid continuous query')
    domain = tuple((max(v[j]-20 for v in poly), min(v[j]+20 for v in poly)) for j in (0, 1))
    if any(lo > hi for lo, hi in domain):
        return {'status': 'empty', 'reason': 'disjoint necessary coordinate intervals'}
    if any(cert.dist2(a, b) > 1600 for a in poly for b in poly):
        return {'status': 'empty', 'reason': 'diameter witness exceeds 40 m'}
    def feasible(q):
        return all(cert.dist2(q, v) <= 400 for v in poly)
    def lower(box):
        return (cert.sqrt_bounds(point_box_distance2(start, box))[0]
                +(cert.sqrt_bounds(point_box_distance2(following, box))[0] if following is not None else 0))/5
    def upper(q):
        return (cert.sqrt_bounds(cert.dist2(start, q))[1]
                +(cert.sqrt_bounds(cert.dist2(q, following))[1] if following is not None else 0))/5
    candidates = [tuple(map(float, start)), tuple(float((a+b)/2) for a, b in domain)]
    if following is not None:
        candidates.append(tuple(map(float, following)))
    # Center-of-radius is a feasible seed, not the travel objective itself.
    seed = clear_center(poly, max_splits=min(max_splits, 128))
    if seed['guaranteed_clear_actual_command']:
        candidates.append(seed['point'])
    best_q, best = None, None
    for q in candidates:
        if feasible(q) and (best is None or upper(q) < best):
            best_q, best = q, upper(q)
    serial = itertools.count()
    heap = [(lower(domain), next(serial), domain)]
    splits, excluded = 0, 0
    while heap and splits < max_splits:
        if best is not None and best-heap[0][0] <= tolerance_s:
            break
        lb, _, box = heapq.heappop(heap)
        if best is not None and lb > best:
            continue
        axis = max(range(2), key=lambda j: box[j][1]-box[j][0])
        mid = sum(box[axis])/2
        if box[axis][0] == box[axis][1]:
            heapq.heappush(heap, (lb, next(serial), box))
            break
        for limits in ((box[axis][0], mid), (mid, box[axis][1])):
            child = list(box)
            child[axis] = limits
            child = tuple(child)
            if any(point_box_distance2(v, child) > 400 for v in poly):
                excluded += 1
                continue
            q = tuple(float(sum(x)/2) for x in child)
            if feasible(q) and (best is None or upper(q) < best):
                best_q, best = q, upper(q)
            bound = lower(child)
            if best is None or bound <= best:
                heapq.heappush(heap, (bound, next(serial), child))
        splits += 1
    if best is None:
        return {'status': 'undetermined' if heap else 'empty', 'splits': splits,
                'unresolved_boxes': len(heap), 'reason': 'no feasible command found within search'}
    lb = min(best, heap[0][0]) if heap else best
    return {'status': 'feasible', 'point': best_q, 'lower_s': str(lb), 'upper_s': str(best),
            'gap_s': float(best-lb), 'epsilon_optimal_for_clear_travel': best-lb <= tolerance_s,
            'splits': splits, 'excluded_boxes': excluded,
            'scope': 'one guaranteed clear with fixed preceding and following positions'}


def prune_physical_cells(belief, cells, budget=32):
    """Conservative intersection with all kept 4-D boxes; no sampled exclusion."""
    boxes, meta = belief.pave(budget)
    if not boxes:
        raise ValueError('physical evidence inconsistent: no source state remains')
    spatial = [(b[0], b[1]) for _, b, _ in boxes]
    kept = []
    for a in cells:
        bounds = tuple((min(v[j] for v in a['cell']), max(v[j] for v in a['cell'])) for j in (0, 1))
        if any(all(x[0] <= y[1] and y[0] <= x[1] for x, y in zip(bounds, b)) for b in spatial):
            kept.append(a)
    if not kept:
        raise ValueError('no certified source cell remains')
    return kept, meta


@dataclass
class ReviewedController(Controller):
    use_count: bool = True
    use_center: bool = True
    use_discovery_dp: bool = True
    use_physical_queries: bool = False
    q4_pair_only: bool = True
    center_splits: int = 64
    minimum_sources: int = 10
    maximum_sources: int = 16

    def __post_init__(self):
        super().__post_init__()
        self.domains = tuple(frozenset((0, 1)) for _ in range(20))
        self.center_versions = {}
        self.discovery_dirty = True
        self.fault = None
        self.metrics.update(count_exclusions=0, count_forced_present=0, center_queries=0,
                            center_adoptions=0, discovery_dp_queries=0, discovery_dp_adoptions=0,
                            physical_queries=0, physical_cell_exclusions=0, dominated_q4_single_skipped=0)

    def cover_plan(self, belief, bearing, channel):
        cells = super().cover_plan(belief, bearing, channel)
        if self.use_physical_queries:
            kept, _ = prune_physical_cells(belief, cells)
            self.metrics['physical_queries'] += 1
            self.metrics['physical_cell_exclusions'] += len(cells)-len(kept)
            return kept
        return cells

    def update_count(self):
        if not self.use_count:
            return
        domains = list(self.domains)
        for c in self.cleared | set(self.tasks):
            domains[c-1] = domains[c-1] & {1}
        for c in self.excluded:
            domains[c-1] = domains[c-1] & {0}
        contracted = count_supports(domains, self.minimum_sources, self.maximum_sources)
        new_excluded = {c for c, d in enumerate(contracted, 1) if d == {0}}-self.excluded
        self.metrics['count_exclusions'] += len(new_excluded)
        self.metrics['count_forced_present'] += sum(a == {0, 1} and b == {1}
                                                   for a, b in zip(domains, contracted))
        self.domains = contracted
        self.excluded.update(new_excluded)
        self.plan = [a for a in self.plan if a['channel'] not in new_excluded]

    def discovery_dp(self):
        if not self.use_discovery_dp or self.q4 or self.tasks or not self.discovery_dirty:
            return
        self.discovery_dirty = False
        if not self.plan:
            return
        required = [set() for _ in core.POINTS]
        for a in self.plan:
            if a['purpose'] != 'discovery':
                return
            required[core.POINTS.index(a['position'])].add(a['channel'])
        route, meta = core.schedule(required, self.position, self.channel)
        self.metrics['discovery_dp_queries'] += 1
        proposal = [action('measure', core.POINTS[i], c, 'discovery') for i, cs in route for c in cs]
        if difference_bounds(proposal, self.plan, self.position, self.channel)[1] <= 0:
            self.plan = proposal
            self.metrics['discovery_dp_adoptions'] += 1

    def adopt_center(self, channel):
        task = self.tasks[channel]
        if not self.use_center or self.center_versions.get(channel) == task.belief.version:
            return
        self.center_versions[channel] = task.belief.version
        poly = task.belief.poly
        # Cheap necessary condition avoids spending a continuous solve on a strip.
        if any(cert.dist2(a, b) > 1600 for a in poly for b in poly):
            return
        rest = [a for a in self.plan if not (a['kind'] == 'clear' and a['channel'] == channel)]
        following = rest[0]['position'] if rest else None
        result = route_center(poly, self.position, following, self.center_splits)
        self.metrics['center_queries'] += 1
        if result['status'] != 'feasible':
            return
        proposal = [action('clear', result['point'], channel, 'cover', poly)]+rest
        if difference_bounds(proposal, self.plan, self.position, self.channel)[1] <= 0:
            self.plan = proposal
            self.metrics['center_adoptions'] += 1

    def try_measure(self, c):
        if not self.q4 or not self.q4_pair_only:
            return super().try_measure(c)
        task = self.tasks[c]
        if (not self.measurements or task.attempts >= 3 or task.extra >= 3 or
            task.measurement_tried_version == task.belief.version):
            return None
        task.measurement_tried_version = task.belief.version
        task.attempts += 1
        if sum(a['channel'] == c and a['kind'] == 'clear' for a in self.plan) <= 1:
            return None
        # Under the archived unchanged no_signal tail, a single measure costs
        # at least5 s more: triangle inequality for travel and switch metric.
        # This theorem does NOT reject sensing after a different posterior tail.
        self.metrics['dominated_q4_single_skipped'] += 1
        return self.try_pair(c, exact_cost(self.plan, self.position, self.channel, len(self.tasks))[0])

    def next_action(self):
        if self.fault:
            raise ValueError(self.fault)
        if self.pending is not None or self.forced is not None:
            return super().next_action()
        self.discovery_dp()
        if not self.plan:
            return None
        self.reorder()
        if self.plan[0]['kind'] == 'clear':
            self.adopt_center(self.plan[0]['channel'])
        chosen = self.plan[0]
        if chosen['kind'] == 'clear':
            chosen = self.try_measure(chosen['channel']) or chosen
        self.pending = chosen
        return chosen

    def accept(self, observation):
        # Archived accept changes the ledger before validating a numeric bearing.
        # Validate malformed receipts first; contradictions then fail closed.
        if observation.get('kind') == 'direction':
            bearing = observation.get('bearing')
            if isinstance(bearing, bool) or not isinstance(bearing, (int, float)) or not math.isfinite(bearing):
                raise ValueError('invalid direction receipt; state unchanged')
        if self.fault:
            raise ValueError(self.fault)
        a = self.pending
        if a is None:
            raise ValueError('no pending action')
        try:
            super().accept(observation)
            self.update_count()
        except ValueError as exc:
            # Unknown is retryable and does not consume the pending action.
            if observation.get('kind') not in ('unknown', None):
                self.fault = str(exc)
            raise
        if (observation['kind'] in ('success', 'near', 'direction') or
            (a['purpose'] == 'discovery' and not any(x['purpose'] == 'discovery' and x['position'] == a['position'] for x in self.plan))):
            self.discovery_dirty = True

    def completed(self):
        return not self.fault and super().completed()
