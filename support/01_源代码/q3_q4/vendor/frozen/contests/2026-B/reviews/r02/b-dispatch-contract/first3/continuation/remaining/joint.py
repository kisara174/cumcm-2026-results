"""Research Q3/Q4 controller: action-level interleaving with complete plan guard.

No HTTP, deadlines or official simulator. State is explicit, copyable, and
resumable; only confirmed observations mutate it. Search nodes and all remaining
clear cells share one plan. Exact geometry is separate from heuristic ordering.
"""
from dataclasses import dataclass, field
from fractions import Fraction as F
from pathlib import Path
import math
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import core
import improvement
import certified as cert
from physical import Belief
from q4_pair import symmetric_pair


def action(kind, q, channel, purpose, cell=None):
    a = {'kind': kind, 'position': tuple(map(float, q)), 'channel': channel, 'purpose': purpose}
    if cell is not None:
        a['cell'] = cell
    return a


def cost(plan, position, channel, active=0):
    value = 2*active
    for a in plan:
        value += math.dist(position, a['position'])/5
        if a['kind'] == 'measure':
            value += 5+int(channel != a['channel'])
            channel = a['channel']
        else:
            value += 3
        position = a['position']
    return value


def exact_cost(plan, position, channel, active=0):
    lo = hi = F(2*active)
    for a in plan:
        length = cert.sqrt_bounds(cert.dist2(position, a['position']))
        fee = 3
        if a['kind'] == 'measure':
            fee = 5+int(channel != a['channel'])
            channel = a['channel']
        lo += length[0]/5+fee
        hi += length[1]/5+fee
        position = a['position']
    return lo, hi


def promote(plan, index):
    return [plan[index]]+plan[:index]+plan[index+1:]


def difference_bounds(new, old, position, channel):
    """Cancel equal squared-length radicals before bounding; proves exact ties."""
    terms = {}
    fees = 0
    for route, sign in ((new, 1), (old, -1)):
        p, ch = position, channel
        for a in route:
            d2 = cert.dist2(p, a['position'])
            terms[d2] = terms.get(d2, 0)+sign
            if a['kind'] == 'measure':
                fees += sign*(5+int(ch != a['channel']))
                ch = a['channel']
            else:
                fees += sign*3
            p = a['position']
    lo = hi = F(fees)
    for d2, coefficient in terms.items():
        if coefficient:
            a, b = cert.sqrt_bounds(d2)
            lo += min(coefficient*a, coefficient*b)/5
            hi += max(coefficient*a, coefficient*b)/5
    return lo, hi


def travel_delta(plan, index, position):
    if index == 0:
        return F(0), F(0)
    q = plan[index]['position']
    before = plan[index-1]['position']
    old_edges = [(position, plan[0]['position']), (before, q)]
    new_edges = [(position, q), (q, plan[0]['position'])]
    if index+1 < len(plan):
        after = plan[index+1]['position']
        old_edges.append((q, after))
        new_edges.append((before, after))
    lower = upper = F(0)
    for a, b in new_edges:
        lo, hi = cert.sqrt_bounds(cert.dist2(a, b))
        lower += lo
        upper += hi
    for a, b in old_edges:
        lo, hi = cert.sqrt_bounds(cert.dist2(a, b))
        lower -= hi
        upper -= lo
    return lower/5, upper/5


@dataclass
class SourceTask:
    belief: Belief
    bearing: float
    extra: int = 0
    measurement_tried_version: int = -1
    attempts: int = 0


@dataclass
class Controller:
    q4: bool = True
    joint: bool = True
    measurements: bool = True
    failed_pruning: bool = True
    depth: int = 1
    beam: int = 4
    position: tuple = (0., 0.)
    channel: int = 1
    spent: float = 0.
    tasks: dict = field(default_factory=dict)
    cleared: set = field(default_factory=set)
    excluded: set = field(default_factory=set)
    pending: dict = None
    saved_branches: object = None
    forced: object = None
    metrics: dict = field(default_factory=lambda: {'reorders': 0, 'interleavings': 0,
                                                  'measure_guards': 0, 'measure_rejects': 0,
                                                  'pruned_cells': 0, 'certified_covers': 0,
                                                  'max_active': 0, 'ledger_checks': 0})

    def __post_init__(self):
        if self.depth not in (1, 2, 3) or self.beam < 1:
            raise ValueError('invalid lookahead configuration')
        points = cert.q4_grid() if self.q4 else core.POINTS
        self.plan = []
        current = 1
        for q in points:
            channels = [current]+[c for c in range(1, 21) if c != current]
            self.plan.extend(action('measure', q, c, 'discovery') for c in channels)
            current = channels[-1]
        self.initial_search_bound = float(exact_cost(self.plan, self.position, 1)[1])
        self.service_reserve = 0.
        self.history = []
        self.beliefs = {c: Belief(self.q4) for c in range(1, 21)}

    def cover_plan(self, belief, bearing, channel):
        cells, meta = cert.certified_cover(belief.poly, bearing,
                                           belief.failed if self.failed_pruning else ())
        if self.failed_pruning:
            kept = [(center, vertices) for center, vertices in cells if not belief.cell_impossible(vertices)]
            self.metrics['pruned_cells'] += len(cells)-len(kept)
            cells = kept
            if not cells:
                raise ValueError('physical constraints exclude entire source cover')
        self.metrics['certified_covers'] += 1
        return [action('clear', center, channel, 'cover', vertices) for center, vertices in cells]

    def candidates(self, plan, position):
        # Search has a fixed complete order. Clear cells are independent across
        # and within sources; only a successful clear may remove its siblings.
        cs = [i for i, a in enumerate(plan) if a['kind'] == 'clear']
        nearest = sorted(cs, key=lambda i: (math.dist(position, plan[i]['position']), i))[:12]
        first_search = next((i for i, a in enumerate(plan) if a['purpose'] == 'discovery'), 0)
        return list(dict.fromkeys([0, first_search]+[i for i in nearest if i]))

    def reorder(self):
        if not self.joint or len(self.plan) < 2:
            return
        # Beam nodes retain the complete tail. No true source or predicted
        # success is used: deterministic all-fail route proxy + success reserve.
        nodes = [(0., [], self.plan, self.position, self.channel)]
        best_score = cost(self.plan, self.position, self.channel)
        best = self.plan
        for _ in range(self.depth):
            expanded = []
            for prefix_cost, prefix, tail, p, ch in nodes:
                if not tail:
                    continue
                for i in self.candidates(tail, p):
                    route = promote(tail, i)
                    a, rest = route[0], route[1:]
                    step = cost([a], p, ch)
                    ch2 = a['channel'] if a['kind'] == 'measure' else ch
                    score = prefix_cost+step+cost(rest, a['position'], ch2)
                    full = prefix+route
                    if score < best_score-1e-6 or (abs(score-best_score) < 1e-8 and
                        full[0]['purpose'] == 'discovery' and best[0]['purpose'] != 'discovery'):
                        best_score, best = score, full
                    expanded.append((score, prefix_cost+step, prefix+[a], rest, a['position'], ch2))
            nodes = [(v, p, t, q, c) for _, v, p, t, q, c in sorted(expanded, key=lambda n: n[0])[:self.beam]]
        if best is not self.plan:
            # Full exact rational distance bounds verify heuristic improvement.
            if difference_bounds(best, self.plan, self.position, self.channel)[1] <= 0:
                self.plan = best
                self.metrics['reorders'] += 1

    def try_measure(self, c):
        task = self.tasks[c]
        if not self.measurements or task.attempts >= 3 or task.extra >= 3 or task.measurement_tried_version == task.belief.version:
            return None
        task.measurement_tried_version = task.belief.version
        task.attempts += 1
        if sum(a['channel'] == c and a['kind'] == 'clear' for a in self.plan) <= 1:
            return None
        poly = tuple(tuple(map(float, p)) for p in task.belief.poly)
        incumbent = core.cover(poly, self.position, self.position)
        # Fast proposal only. The full rational tree below is authoritative.
        proposal, _ = improvement.choose_measure(poly, self.position, self.position, incumbent,
                                                 max_splits=64)
        if proposal is None:
            if self.q4:
                return self.try_pair(c, exact_cost(self.plan, self.position, self.channel, len(self.tasks))[0])
            return None
        q = proposal[0]
        rest = [a for a in self.plan if not (a['kind'] == 'clear' and a['channel'] == c)]
        branches = {}
        worst = F(0)
        for lo in range(0, 360, 10):
            p = cert.wedge(task.belief.poly, tuple(map(cert.rat, q)), F(lo+5), F(5)+task.belief.report_half)
            if p:
                cells, _ = cert.certified_cover(p, lo+5, task.belief.failed if self.failed_pruning else (), allow_empty=True)
                if not cells:
                    continue
                tail = [action('clear', center, c, 'cover', vertices) for center, vertices in cells]+rest
                branches[('direction', lo//10)] = tail
        near = cert.rectangle_clip(task.belief.poly, tuple(map(cert.rat, q)), 5)
        if near:
            cells, _ = cert.certified_cover(near, 0)
            branches[('near', 0)] = [action('clear', center, c, 'cover', vertices) for center, vertices in cells]+rest
        # Always retain no_signal for Q4: a nearby point can be behind emitter.
        if self.q4 or any(cert.dist2(p, q) > 1000**2 for p in task.belief.poly):
            branches[('no_signal', 0)] = list(self.plan)
        m = action('measure', q, c, 'localize')
        for tail in branches.values():
            worst = max(worst, exact_cost([m]+tail, self.position, self.channel, len(self.tasks))[1])
        current_lower = exact_cost(self.plan, self.position, self.channel, len(self.tasks))[0]
        if branches and worst < current_lower:
            self.saved_branches = branches
            self.metrics['measure_guards'] += 1
            return m
        self.metrics['measure_rejects'] += 1
        if self.q4:
            return self.try_pair(c, current_lower)
        return None

    def try_pair(self, c, current_lower):
        task = self.tasks[c]
        positive = next((q for q, o, a in task.belief.events if a == 'measure' and o['kind'] in ('near', 'direction')), None)
        if positive is None or task.extra > 1:
            return None
        rest = [a for a in self.plan if not (a['kind'] == 'clear' and a['channel'] == c)]
        for offset in ((0, 200), (200, 0), (0, 100), (100, 0)):
            pair = symmetric_pair(task.belief.poly, positive, offset)
            if not pair['certified']:
                continue
            def positive_tails(q):
                tails = {}
                for lo in range(0, 360, 2):
                    p = cert.wedge(task.belief.poly, tuple(map(cert.rat, q)), F(lo+1), F(1)+task.belief.report_half)
                    if not p:
                        continue
                    cells, _ = cert.certified_cover(p, lo+1, task.belief.failed, allow_empty=True)
                    if cells:
                        tails[('direction', lo, lo+2)] = [action('clear', center, c, 'cover', vertices) for center, vertices in cells]+rest
                p = cert.rectangle_clip(task.belief.poly, tuple(map(cert.rat, q)), 5)
                if p:
                    cells, _ = cert.certified_cover(p, 0, task.belief.failed, allow_empty=True)
                    if cells:
                        tails[('near', 0)] = [action('clear', center, c, 'cover', vertices) for center, vertices in cells]+rest
                return tails
            first = action('measure', pair['plus'], c, 'localize')
            second = action('measure', pair['minus'], c, 'localize')
            first_tails = positive_tails(pair['plus'])
            second_tails = positive_tails(pair['minus'])
            if not first_tails or not second_tails:
                continue
            values = [exact_cost([first]+tail, self.position, self.channel, len(self.tasks))[1] for tail in first_tails.values()]
            values += [exact_cost([first, second]+tail, self.position, self.channel, len(self.tasks))[1] for tail in second_tails.values()]
            if max(values) < current_lower:
                first_tails[('no_signal', 0)] = {'action': second, 'branches': second_tails}
                self.saved_branches = first_tails
                self.metrics['paired_measure_guards'] = self.metrics.get('paired_measure_guards', 0)+1
                return first
        return None

    def next_action(self):
        if self.pending is not None:
            return self.pending  # Caller must not create a new action before receipt.
        if self.forced is not None:
            self.pending = self.forced['action']
            self.saved_branches = self.forced['branches']
            self.forced = None
            return self.pending
        if not self.plan:
            return None
        self.reorder()
        chosen = self.plan[0]
        if chosen['kind'] == 'clear':
            chosen = self.try_measure(chosen['channel']) or chosen
        self.pending = chosen
        return chosen

    def accept(self, observation):
        if self.pending is None:
            raise ValueError('no pending action')
        a = self.pending
        if observation.get('kind') not in (('near', 'direction', 'no_signal') if a['kind'] == 'measure' else ('success', 'no_target_in_range')):
            raise ValueError('unknown is not confirmed feedback')
        c, q = a['channel'], a['position']
        self.spent += cost([a], self.position, self.channel)+(2 if observation['kind'] == 'success' else 0)
        if a['kind'] == 'measure':
            self.channel = c
        if self.history and self.history[-1]['channel'] != c and c in self.tasks:
            self.metrics['interleavings'] += 1
        self.position = q
        if a['purpose'] == 'localize':
            if observation['kind'] == 'direction':
                bearing = observation['bearing'] % 360
                matches = [k for k in self.saved_branches if k[0] == 'direction' and
                           ((len(k) == 2 and k[1] == int(bearing//10)) or
                            (len(k) == 3 and k[1] <= bearing < k[2]))]
                if len(matches) != 1:
                    raise ValueError('direction not uniquely covered')
                key = matches[0]
            else:
                key = (observation['kind'], 0)
            if key not in self.saved_branches:
                raise ValueError('observation outside saved tree')
            child = self.saved_branches[key]
            if isinstance(child, dict):
                self.forced = child
            else:
                self.plan = child
            task = self.tasks[c]
            task.belief.observe(q, observation)
            task.extra += 1
            self.saved_branches = None
        else:
            assert self.plan[0] == a
            self.plan.pop(0)
            if a['purpose'] == 'discovery':
                belief = self.beliefs[c]
                belief.observe(q, observation)
                if observation['kind'] == 'no_signal':
                    if not any(x['channel'] == c for x in self.plan):
                        self.excluded.add(c)
                else:
                    bearing = observation.get('bearing', 0.)
                    task = SourceTask(belief, bearing)
                    self.tasks[c] = task
                    self.plan = [x for x in self.plan if x['channel'] != c]
                    clear = self.cover_plan(belief, bearing, c)
                    reserve = exact_cost(clear+[action('measure', q, c, 'accounting-only')], q, c, 1)[1]
                    self.service_reserve += float(reserve)
                    self.plan = clear+self.plan
            else:
                task = self.tasks[c]
                task.belief.observe(q, observation, 'clear')
                if observation['kind'] == 'success':
                    self.cleared.add(c)
                    del self.tasks[c]
                    self.plan = [x for x in self.plan if x['channel'] != c]
                elif self.failed_pruning:
                    retained = []
                    for x in self.plan:
                        if x['kind'] == 'clear' and x['channel'] == c and all(cert.dist2(p, q) <= 400 for p in x['cell']):
                            self.metrics['pruned_cells'] += 1
                        else:
                            retained.append(x)
                    self.plan = retained
        self.history.append({'kind': a['kind'], 'position': q, 'channel': c,
                             'purpose': a['purpose'], 'observation': dict(observation)})
        self.pending = None
        self.metrics['max_active'] = max(self.metrics['max_active'], len(self.tasks))
        self.metrics['ledger_checks'] += 1
        remaining = cost(self.plan, self.position, self.channel, len(self.tasks))
        if self.forced is not None:
            remaining = max(float(exact_cost([self.forced['action']]+tail, self.position, self.channel, len(self.tasks))[1])
                            for tail in self.forced['branches'].values())
        ledger = self.spent+remaining
        assert ledger <= self.initial_search_bound+self.service_reserve+1e-4

    def completed(self):
        return self.pending is None and self.forced is None and not self.plan and not self.tasks and self.cleared | self.excluded == set(range(1, 21))
