"""Q3 improvements; the c2bb336 core and evidence remain unchanged.

The anchor is an accounting potential, not a fictitious executed movement.
Feedback refinement is atomic and retains a complete parent cover on timeout.
Ordinary floating geometry: not an interval certificate or a global optimizer.
"""
from dataclasses import dataclass
import math
import time

import core as base


@dataclass(frozen=True)
class Cell:
    kind: str
    lo: float
    hi: float
    poly: tuple
    plan: base.Cover


def check_deadline(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError


def coarse_tree(poly, q, anchor, deadline=math.inf):
    """Same complete 10-degree feedback enclosure as the archived baseline."""
    cells = []
    for lo in range(0, 360, 10):
        check_deadline(deadline)
        p = base.wedge(poly, q, lo+5, 5+base.ERROR)
        if p:
            cells.append(Cell('direction', lo, lo+10, p, base.cover(p, q, anchor)))
    check_deadline(deadline)
    p = base.in_square(poly, q, 5)
    if p:
        cells.append(Cell('near', 0, 0, p, base.cover(p, q, anchor)))
    check_deadline(deadline)
    if max(math.dist(q, p) for p in poly) > 1000-1e-5:
        cells.append(Cell('no_signal', 0, 0, poly, base.cover(poly, q, anchor)))
    check_deadline(deadline)
    if not cells:
        raise ValueError('empty feedback tree')
    return tuple(cells)


def tree_bound(cells):
    return max(c.plan.bound_s for c in cells)


def response_plan(cells, obs):
    kind = obs.get('kind')
    if kind == 'direction':
        bearing = obs.get('bearing')
        if not isinstance(bearing, (int, float)) or not math.isfinite(bearing):
            raise ValueError('invalid bearing')
        bearing %= 360
        matches = [c for c in cells if c.kind == kind and c.lo <= bearing < c.hi]
    else:
        matches = [c for c in cells if c.kind == kind]
    if len(matches) != 1:
        raise ValueError('feedback not covered uniquely by saved tree')
    return matches[0].plan


def refine_once(cells, q, anchor, min_width_deg=1., deadline=math.inf):
    """Split a worst-cost cell; both children complete before replacing parent.

Stop when a maximizing leaf is non-direction or already at resolution: no
other split can lower the current maximum. This is not an optimality gap.
"""
    if not 0 < min_width_deg <= 10:
        raise ValueError('min_width_deg must lie in (0, 10]')
    check_deadline(deadline)
    i = max(range(len(cells)), key=lambda i: cells[i].plan.bound_s)
    parent = cells[i]
    if parent.kind != 'direction' or parent.hi-parent.lo <= min_width_deg:
        return cells, False
    mid = (parent.lo+parent.hi)/2
    children = []
    for lo, hi in ((parent.lo, mid), (mid, parent.hi)):
        check_deadline(deadline)
        p = base.wedge(parent.poly, q, (lo+hi)/2, (hi-lo)/2+base.ERROR)
        if p:
            fresh = base.cover(p, q, anchor)
            plan = fresh if fresh.bound_s <= parent.plan.bound_s else parent.plan
            children.append(Cell('direction', lo, hi, p, plan))
    check_deadline(deadline)
    if not children:
        raise ValueError('nonempty parent lost both feedback children')
    result = cells[:i]+tuple(children)+cells[i+1:]
    assert tree_bound(result) <= tree_bound(cells)+1e-8
    return result, True


def choose_measure(poly, position, anchor, incumbent, group='union',
                   budget_s=math.inf, refine=True, min_width_deg=1.,
                   max_splits=512):
    """Coarse pass over candidates, then round-robin worst-leaf refinement.

An interrupted coarse candidate is discarded. An interrupted refinement keeps
its old full tree. A coarse loser is never pruned by its upper bound.
"""
    if math.isnan(budget_s) or budget_s < 0 or max_splits < 0:
        raise ValueError('invalid planning budget')
    if not 0 < min_width_deg <= 10:
        raise ValueError('invalid feedback resolution')
    start = time.perf_counter()
    deadline = start+budget_s
    candidates = []
    selected = None
    best = incumbent.bound_s
    splits = 0
    timed_out = False

    def consider(q, cells):
        nonlocal best, selected
        score = math.dist(position, q)/5+5+tree_bound(cells)
        if score < best-1e-6:
            best = score
            selected = (q, cells, score)

    for q in base.candidates(poly, position, group):
        try:
            cells = coarse_tree(poly, q, anchor, deadline)
        except TimeoutError:
            timed_out = True
            break
        candidates.append([q, cells, True])
        consider(q, cells)
    coarse_best = best
    if refine and not timed_out:
        try:
            while splits < max_splits and any(item[2] for item in candidates):
                for item in candidates:
                    if not item[2] or splits >= max_splits:
                        continue
                    cells, changed = refine_once(item[1], item[0], anchor,
                                                 min_width_deg, deadline)
                    item[1], item[2] = cells, changed
                    if changed:
                        splits += 1
                        consider(item[0], cells)
        except TimeoutError:
            timed_out = True
    assert best <= coarse_best+1e-8
    return selected, {'evaluated': len(candidates), 'splits': splits,
                      'timed_out': timed_out, 'wall_s': time.perf_counter()-start,
                      'old_bound_s': incumbent.bound_s, 'coarse_bound_s': coarse_best,
                      'selected_bound_s': best,
                      'split_limit_reached': refine and splits >= max_splits,
                      'selected_cells': len(selected[1]) if selected else 0}


def localize(anchor, channel, first, group='union', budget_s=math.inf,
             refine=True, return_to_anchor=False, max_extra=3):
    poly = base.initial_polygon(anchor, first)
    position = anchor
    incumbent = base.initial_cover(anchor, first)
    initial_bound = incumbent.bound_s
    spent = 0.
    decisions = []
    for _ in range(max_extra if group != 'none' else 0):
        selected, meta = choose_measure(poly, position, anchor, incumbent, group,
                                        budget_s, refine)
        decisions.append(meta)
        if selected is None:
            break
        q, cells, _ = selected
        obs = yield {'kind': 'measure', 'position': q, 'channel': channel,
                     'purpose': 'localize', 'enclosure': poly}
        inherited = response_plan(cells, obs)
        spent += math.dist(position, q)/5+5
        poly = base.posterior(poly, q, obs)
        fresh = base.cover(poly, q, anchor)
        incumbent = fresh if fresh.bound_s <= inherited.bound_s else inherited
        position = q
        assert spent+incumbent.bound_s <= initial_bound+1e-5
    for point in incumbent.points:
        obs = yield {'kind': 'clear', 'position': point, 'channel': channel,
                     'purpose': 'cover', 'enclosure': poly}
        if obs['kind'] not in ('success', 'no_target_in_range'):
            raise ValueError('unconfirmed clear cannot advance fallback')
        spent += math.dist(position, point)/5+(5 if obs['kind'] == 'success' else 3)
        position = point
        if obs['kind'] == 'success':
            break
    else:
        raise ValueError('finite cover exhausted without success')
    # This is a potential only. No return receipt is fabricated or credited.
    exit_credit = math.dist(position, anchor)/5+5
    assert spent+exit_credit <= initial_bound+1e-5
    if return_to_anchor:  # Factorial experimental control, not the default.
        obs = yield {'kind': 'measure', 'position': anchor, 'channel': channel,
                     'purpose': 'return'}
        if obs['kind'] != 'no_signal':
            raise ValueError('unexpected return response')
        spent += exit_credit
        position, exit_credit = anchor, 0.
    return {'initial_bound_s': initial_bound, 'actual_s': spent,
            'position': position, 'anchor': anchor, 'exit_credit_s': exit_credit,
            'decisions': decisions}


def controller(mode='exact', group='union', budget_s=math.inf,
               refine=True, return_to_anchor=False):
    required = [set(range(1, 21)) for _ in base.POINTS]
    cleared, excluded = set(), set()
    position, channel = (0., 0.), 1
    cached_route = None
    virtual_s = 0.
    metrics = {'planning': [], 'sources': [], 'responsibility_history': [],
               'global_ledger_s': [], 'planning_positions': []}
    while any(required):
        route, meta = base.schedule(required, position, channel, mode=mode,
                                    budget_s=budget_s, incumbent_route=cached_route)
        cached_route = list(route)
        ledger = virtual_s+base.route_cost(route, base.POINTS, position, channel)+(16-len(cleared))*1746
        saved_measurements = 0 if return_to_anchor else 5*len(cleared)
        assert ledger <= 30569-saved_measurements+1e-5
        metrics['global_ledger_s'].append(ledger)
        metrics['planning_positions'].append(position)
        metrics['planning'].append(meta)
        i, order = route[0]
        discovered = False
        for c in order:
            obs = yield {'kind': 'measure', 'position': base.POINTS[i], 'channel': c,
                         'purpose': 'discovery'}
            if obs['kind'] not in ('direction', 'near', 'no_signal'):
                raise ValueError('unconfirmed discovery receipt')
            virtual_s += math.dist(position, base.POINTS[i])/5+5+int(c != channel)
            position, channel = base.POINTS[i], c
            required[i].remove(c)
            cached_route[0] = (i, tuple(x for x in cached_route[0][1] if x != c))
            if not cached_route[0][1]:
                cached_route.pop(0)
            if obs['kind'] in ('direction', 'near'):
                for duties in required:
                    duties.discard(c)
                info = yield from localize(position, c, obs, group, budget_s,
                                            refine, return_to_anchor)
                metrics['sources'].append(info)
                virtual_s += info['actual_s']
                position = info['position']  # Actual successful clear point.
                cleared.add(c)
                discovered = True
            elif not any(c in duties for duties in required):
                excluded.add(c)
            metrics['responsibility_history'].append(sum(map(len, required)))
            if discovered:
                break
    assert cleared.isdisjoint(excluded) and cleared | excluded == set(range(1, 21))
    assert virtual_s <= 30569-(0 if return_to_anchor else 5*len(cleared))+1e-5
    return {'completed': True, 'controller_virtual_s': virtual_s,
            'cleared_channels': sorted(cleared), 'excluded_channels': sorted(excluded),
            **metrics}


def run(generator, env, max_actions=5000):
    """Audit actual feedback/time. Works with archived and improved controllers."""
    observed_costs = []
    local_start = None
    pending = False
    try:
        action = next(generator)
        for _ in range(max_actions):
            if action['purpose'] == 'discovery' and pending:
                observed_costs.append(env.time_s-local_start)
                pending = False
            obs = env.step(action)
            if action['purpose'] == 'discovery' and obs['kind'] in ('near', 'direction'):
                local_start, pending = env.time_s, True
            action = generator.send(obs)
    except StopIteration as end:
        if pending:
            observed_costs.append(env.time_s-local_start)
        result = end.value
        assert result['completed']
        assert set(result['cleared_channels']) == set(env.sources) == env.cleared
        assert set(result['excluded_channels']).isdisjoint(env.sources)
        assert abs(result['controller_virtual_s']-env.time_s) < 1e-5
        assert len(observed_costs) == len(result['sources'])
        for actual, source in zip(observed_costs, result['sources']):
            assert abs(actual-source['actual_s']) < 1e-5
            assert actual+source.get('exit_credit_s', 0) <= source['initial_bound_s']+1e-5
        result.update(total_s=env.time_s, actions=len(env.trace), trace=env.trace)
        for field, key, value in [('clear_calls', 'kind', 'clear'),
                                   ('return_calls', 'purpose', 'return'),
                                   ('extra_measure_calls', 'purpose', 'localize'),
                                   ('discovery_calls', 'purpose', 'discovery')]:
            result[field] = sum(row[key] == value for row in env.trace)
        return result
    raise AssertionError('action limit exhausted')
