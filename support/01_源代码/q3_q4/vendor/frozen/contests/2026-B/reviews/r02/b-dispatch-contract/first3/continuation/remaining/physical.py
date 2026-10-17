"""Joint x,y,R,orientation,type constraints and sound box set inversion.

The constraint history defines the nonconvex physical set implicitly. Finite
box paving is an OUTER approximation, not an exact finite polygon solution.
"""
from dataclasses import dataclass, field
from fractions import Fraction as F
import math

import certified as cert


def interval(x):
    return cert.rat(x), cert.rat(x)


def sub(a, b):
    return cert.add(a, cert.neg(b))


def sq(a):
    return ((F(0) if a[0] <= 0 <= a[1] else min(a[0]**2, a[1]**2)),
            max(a[0]**2, a[1]**2))


def le(a, b=(F(0), F(0))):
    if a[0] > b[1]:
        return -1
    return 1 if a[1] <= b[0] else 0


def gt(a, b=(F(0), F(0))):
    return -le(a, b)


def conjunction(values):
    return min(values)


def disjunction(values):
    return max(values)


def trig_range(a):
    mid = (a[0]+a[1])/2
    s, c = cert.trig(mid)
    radius = (a[1]-a[0])/2*cert.pi_interval()[1]/180
    return tuple((max(F(-1), t[0]-radius), min(F(1), t[1]+radius)) for t in (s, c))


@dataclass
class Belief:
    report_half = F(201, 200)
    q4: bool = True
    events: list = field(default_factory=list)
    measured: dict = field(default_factory=dict)
    poly: tuple = field(default_factory=cert.square)
    failed: list = field(default_factory=list)
    version: int = 0
    inconsistent: bool = False

    def observe(self, q, obs, action='measure'):
        q = tuple(map(cert.rat, q))
        kind = obs.get('kind')
        allowed = ('direction', 'near', 'no_signal') if action == 'measure' else ('success', 'no_target_in_range')
        if kind not in allowed:
            raise ValueError('unknown receipt is not physical evidence')
        if kind == 'direction' and not math.isfinite(obs.get('bearing', math.nan)):
            raise ValueError('invalid direction')
        if action == 'measure':
            signature = (kind, F(str(obs['bearing'])) if kind == 'direction' else None)
            if q in self.measured and self.measured[q] != signature:
                self.inconsistent = True
                raise ValueError('spatially fixed measurement contradiction')
            self.measured[q] = signature
        self.events.append((q, dict(obs), action))
        if kind in ('near', 'direction'):
            self.poly = cert.positive_polygon(self.poly, q, obs)
        if kind == 'no_target_in_range':
            self.failed.append(q)
        self.version += 1
        if not self.poly:
            self.inconsistent = True
            raise ValueError('empty certified enclosure')

    def classify(self, box, directional):
        """-1 infeasible everywhere, 0 unresolved, +1 feasible everywhere."""
        if self.inconsistent:
            return -1
        x, y, radius, angle = box
        values = [le(cert.add(sq(x), sq(y)), interval(1800**2))]
        sin, cos = trig_range(angle) if directional else (None, None)
        for q, obs, action in self.events:
            dx, dy = sub(interval(q[0]), x), sub(interval(q[1]), y)
            d2 = cert.add(sq(dx), sq(dy))
            if action == 'clear':
                values.append(gt(d2, interval(400)) if obs['kind'] == 'no_target_in_range' else le(d2, interval(400)))
                continue
            distance_ok = le(d2, sq(radius))
            visibility = cert.add(cert.mul(cos, dx), cert.mul(sin, dy)) if directional else interval(1)
            visible = le(cert.neg(visibility))
            received = min(distance_ok, visible)
            if obs['kind'] == 'no_signal':
                values.append(-received)
            else:
                values.append(received)
                values.append(le(d2, interval(25)) if obs['kind'] == 'near' else gt(d2, interval(25)))
                if obs['kind'] == 'direction':
                    angle_value = F(str(obs['bearing']))
                    ls, lc = cert.trig(angle_value-self.report_half)
                    us, uc = cert.trig(angle_value+self.report_half)
                    # Vector from detector to source is -(dx,dy).
                    values.append(le(cert.add(cert.mul(cert.neg(ls), dx), cert.mul(lc, dy))))
                    values.append(le(cert.add(cert.mul(us, dx), cert.mul(cert.neg(uc), dy))))
            if -1 in values:
                return -1
        return min(values)

    def roots(self):
        x = (min(p[0] for p in self.poly), max(p[0] for p in self.poly))
        y = (min(p[1] for p in self.poly), max(p[1] for p in self.poly))
        return [(False, (x, y, (F(1000), F(1500)), (F(0), F(0))))]+(
            [(True, (x, y, (F(1000), F(1500)), (F(0), F(360))))] if self.q4 else [])

    def pave(self, max_splits=128, widths=(25, 25, 25, 5)):
        """Every untouched box remains in output when finite budget is exhausted."""
        pending = self.roots()
        kept, rejected, splits = [], 0, 0
        while pending:
            kind, box = pending.pop()
            state = self.classify(box, kind)
            if state == -1:
                rejected += 1
                continue
            axis = max(range(4), key=lambda j: (box[j][1]-box[j][0])/widths[j])
            if state == 1 or box[axis][1]-box[axis][0] <= widths[axis] or splits >= max_splits:
                kept.append((kind, box, state))
                continue
            mid = sum(box[axis])/2
            for limits in ((box[axis][0], mid), (mid, box[axis][1])):
                child = list(box)
                child[axis] = limits
                pending.append((kind, tuple(child)))
            splits += 1
        return kept, {'splits': splits, 'rejected': rejected, 'remaining': len(kept),
                      'unresolved': sum(s == 0 for _, _, s in kept),
                      'exact_set_solved': all(s == 1 for _, _, s in kept),
                      'certified_outer_enclosure': True}

    def truth_not_rejected(self, point, radius, direction=None):
        box = tuple(interval(z) for z in (*point, radius, direction or 0))
        return self.classify(box, direction is not None) != -1

    def cell_impossible(self, vertices):
        """Cheap exact spatial projections; directional negatives stay coupled."""
        x = (min(v[0] for v in vertices), max(v[0] for v in vertices))
        y = (min(v[1] for v in vertices), max(v[1] for v in vertices))
        if cert.add(sq(x), sq(y))[0] > 1800**2:
            return True
        for q, obs, action in self.events:
            d2 = cert.add(sq(sub(x, interval(q[0]))), sq(sub(y, interval(q[1]))))
            if action == 'clear' and obs['kind'] == 'no_target_in_range' and d2[1] <= 400:
                return True
            if action == 'measure':
                if obs['kind'] == 'no_signal' and not self.q4 and d2[1] <= 1000**2:
                    return True
                if obs['kind'] in ('near', 'direction') and d2[0] > 1500**2:
                    return True
                if obs['kind'] == 'near' and d2[0] > 25:
                    return True
        return False
