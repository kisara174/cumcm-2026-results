"""Q3 research prototype. Analytic enclosures, ordinary floating point, no HTTP."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import time

ERROR = 1.01  # includes two-decimal reported-bearing rounding
EPS = 1e-7
POINTS = ((0., 0.),) + tuple((1500*math.cos(k*math.pi/3),
                             1500*math.sin(k*math.pi/3)) for k in range(6))


def dot(a, b):
    return a[0]*b[0] + a[1]*b[1]


def axes(angle):
    t = math.radians(angle)
    return (math.cos(t), math.sin(t)), (-math.sin(t), math.cos(t))


def world(x, y, u, v):
    return (x*u[0]+y*v[0], x*u[1]+y*v[1])


def clip(poly, normal, offset):
    """Clip convex polygon by n.x <= b with outward numerical slack."""
    if not poly:
        return ()
    result = []
    a = poly[-1]
    fa = dot(normal, a) - offset - EPS
    for b in poly:
        fb = dot(normal, b) - offset - EPS
        if (fa <= 0) != (fb <= 0):
            t = fa/(fa-fb)
            result.append((a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])))
        if fb <= 0:
            result.append(b)
        a, fa = b, fb
    return tuple(result)


def square(q, radius):
    x, y = q
    return ((x-radius, y-radius), (x+radius, y-radius),
            (x+radius, y+radius), (x-radius, y+radius))


def in_square(poly, q, radius):
    for n in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        poly = clip(poly, n, dot(n, q)+radius)
    return poly


def wedge(poly, q, bearing, half=ERROR):
    if not 0 < half < 90:
        raise ValueError('wedge half angle must lie in (0, 90)')
    lower, _ = axes(bearing-half)
    upper, _ = axes(bearing+half)
    for n in ((lower[1], -lower[0]), (-upper[1], upper[0])):
        poly = clip(poly, n, dot(n, q))
    return poly


def initial_polygon(q, observation):
    if observation['kind'] == 'near':
        return square(q, 5)
    if observation['kind'] != 'direction':
        raise ValueError('initialization requires a confirmed positive observation')
    u, v = axes(observation['bearing'])
    # Projection bounds valid for distance <= 1500 and angular error <= ERROR.
    s = 1500*math.sin(math.radians(ERROR)) + EPS
    x, y = dot(q, u), dot(q, v)
    poly = tuple(world(a, b, u, v) for a, b in
                 ((x-EPS, y-s), (x+1500+EPS, y-s),
                  (x+1500+EPS, y+s), (x-EPS, y+s)))
    return in_square(wedge(poly, q, observation['bearing']), (0, 0), 1800)


def posterior(poly, q, observation):
    kind = observation['kind']
    if kind == 'direction':
        return in_square(wedge(poly, q, observation['bearing']), q, 1500)
    if kind == 'near':
        return in_square(poly, q, 5)
    if kind == 'no_signal':
        # Safe but loose: do not claim to implement circular exclusion.
        return poly
    raise ValueError('unknown/rejected receipts are not observations')


def contains(poly, point, tol=1e-5):
    return bool(poly) and all((b[0]-a[0])*(point[1]-a[1]) -
                             (b[1]-a[1])*(point[0]-a[0]) >= -tol
                             for a, b in zip(poly, poly[1:]+poly[:1]))


def principal(poly):
    if not poly:
        raise ValueError('empty feasible enclosure')
    a, b = max(((a, b) for a in poly for b in poly), key=lambda ab: math.dist(*ab))
    return math.degrees(math.atan2(b[1]-a[1], b[0]-a[0])) % 180


@dataclass(frozen=True)
class Cover:
    points: tuple
    bound_s: float
    cell_radius_m: float


def cover_cost(points, start, anchor):
    # Execute until first success, then /measure at anchor (same channel).
    path = (start,) + tuple(points) + (anchor,)
    return sum(math.dist(a, b) for a, b in zip(path, path[1:]))/5 + 3*len(points)+2+5


def cover(poly, start, anchor):
    """Cover the WHOLE rotated bounding rectangle, not sampled feasible points."""
    if not poly:
        raise ValueError('empty enclosure cannot be a completed source')
    best = None
    for angle in (principal(poly), 0.):
        u, v = axes(angle)
        xs, ys = [dot(p, u) for p in poly], [dot(p, v) for p in poly]
        lo, hi, bot, top = min(xs)-EPS, max(xs)+EPS, min(ys)-EPS, max(ys)+EPS
        w, h = hi-lo, top-bot
        center = world((lo+hi)/2, (bot+top)/2, u, v)
        radius = math.hypot(w, h)/2
        if radius < 20-1e-5:
            plan = Cover((center,), cover_cost((center,), start, anchor), radius)
            if best is None or plan.bound_s < best.bound_s:
                best = plan
            continue
        nx, ny = max(1, math.ceil(w/25)), max(1, math.ceil(h/25))
        if nx*ny > 1000:
            continue
        # Four snake orientations; ordering ties have explicit stable IDs.
        for flipx in (False, True):
            for flipy in (False, True):
                cells = []
                for i in range(nx):
                    ii = nx-1-i if flipx else i
                    for j in range(ny):
                        jj = ny-1-j if (bool(i % 2) != flipy) else j
                        cells.append(world(lo+(ii+.5)*w/nx, bot+(jj+.5)*h/ny, u, v))
                cost = cover_cost(cells, start, anchor)
                if best is None or cost < best.bound_s-1e-9:
                    best = Cover(tuple(cells), cost, math.hypot(w/nx, h/ny)/2)
    if best is None:
        raise ValueError('enclosure outside intended detected-source strip scope')
    return best


def initial_cover(anchor, first):
    poly = initial_polygon(anchor, first)
    generic = cover(poly, anchor, anchor)
    if first['kind'] == 'near':
        return generic
    u, v = axes(first['bearing'])
    cells = []
    for k in range(60):
        for y in ((-25, 0, 25) if k % 2 == 0 else (25, 0, -25)):
            p = world(12.5+25*k, y, u, v)
            cells.append((anchor[0]+p[0], anchor[1]+p[1]))
    strip = Cover(tuple(cells), cover_cost(cells, anchor, anchor), math.sqrt(2)*12.5)
    return generic if generic.bound_s < strip.bound_s else strip


def candidates(poly, position, group='union'):
    """Geometry candidates scale with current multi-observation enclosure."""
    u, v = axes(principal(poly))
    xs, ys = [dot(p, u) for p in poly], [dot(p, v) for p in poly]
    lo, hi, mid = min(xs), max(xs), (min(ys)+max(ys))/2
    half = (hi-lo)/2
    geometric = [world(lo+f*(hi-lo), mid+s*scale*max(half, 5), u, v)
                 for f in (.5, .25, .75) for scale in (1., .5) for s in (-1, 1)]
    fixed = [(position[0]+r*(math.cos(math.radians(a))*u[0]+s*math.sin(math.radians(a))*v[0]),
              position[1]+r*(math.cos(math.radians(a))*u[1]+s*math.sin(math.radians(a))*v[1]))
             for r in (50, 100, 200, 400, 800) for a in (30, 60) for s in (-1, 1)]
    interleaved = [item for i in range(max(len(fixed), len(geometric)))
                   for family in (geometric, fixed) for item in family[i:i+1]]
    pool = fixed if group == 'fixed' else geometric if group == 'geometry' else interleaved
    # Quantized coordinates define the ACTUAL candidate, not just its sort key.
    return tuple(dict.fromkeys((round(p[0], 6), round(p[1], 6)) for p in pool
                              if math.dist(p, position) > 1e-4))


def branches(poly, q, anchor, bin_deg=10, deadline=math.inf):
    """Every direction reading belongs to a cell; clip by the WHOLE cell + error."""
    if 360 % bin_deg or not 0 < bin_deg < 90:
        raise ValueError('bin width must divide 360 and be below 90')
    result = {}
    for index in range(int(360/bin_deg)):
        if time.perf_counter() >= deadline:
            raise TimeoutError
        p = wedge(poly, q, (index+.5)*bin_deg, bin_deg/2+ERROR)
        if p:
            result[('direction', index)] = cover(p, q, anchor)
    p = in_square(poly, q, 5)
    if p:
        result[('near', 0)] = cover(p, q, anchor)
    # For Q3 R >= 1000, max distance at vertices bounds the entire convex poly.
    # If not guaranteed received, retain unchanged K in a no_signal branch.
    if max(math.dist(q, p) for p in poly) > 1000-1e-5:
        result[('no_signal', 0)] = cover(poly, q, anchor)
    return result


def choose_measure(poly, position, anchor, incumbent, group='union',
                   bin_deg=10, budget_s=math.inf):
    begin = time.perf_counter()
    deadline = begin+budget_s
    selected = None
    best = incumbent.bound_s
    evaluated = 0
    timed_out = False
    for q in candidates(poly, position, group):
        if time.perf_counter() >= deadline:
            timed_out = True
            break
        try:
            children = branches(poly, q, anchor, bin_deg, deadline)
        except TimeoutError:
            timed_out = True
            break  # Never use a partially evaluated feedback tree.
        if time.perf_counter() >= deadline:
            timed_out = True
            break
        evaluated += 1
        value = math.dist(position, q)/5+5+max(c.bound_s for c in children.values())
        if value < best-1e-6:
            best = value
            selected = (q, children, value)
    return selected, {'evaluated': evaluated, 'timed_out': timed_out,
                      'wall_s': time.perf_counter()-begin, 'old_bound_s': incumbent.bound_s,
                      'selected_bound_s': best}


def localize(anchor, channel, first, group='union', budget_s=math.inf, max_extra=3):
    """Generator consumes only feedback. It has no scenario/hidden truth argument."""
    poly = initial_polygon(anchor, first)
    position = anchor
    incumbent = initial_cover(anchor, first)
    initial_bound = incumbent.bound_s
    spent = 0.
    decisions = []
    for _ in range(max_extra if group != 'none' else 0):
        selected, meta = choose_measure(poly, position, anchor, incumbent, group,
                                        budget_s=budget_s)
        decisions.append(meta)
        if selected is None:
            break
        q, children, value = selected
        obs = yield {'kind': 'measure', 'position': q, 'channel': channel,
                     'purpose': 'localize', 'enclosure': poly}
        spent += math.dist(position, q)/5+5
        key = ('direction', int((obs['bearing'] % 360)//10)) if obs['kind'] == 'direction' else (obs['kind'], 0)
        if key not in children:
            raise ValueError('feedback not covered by predicted branches')
        poly = posterior(poly, q, obs)
        inherited = children[key]  # Keep the exact certified continuation, not its scalar only.
        fresh = cover(poly, q, anchor)
        incumbent = fresh if fresh.bound_s <= inherited.bound_s else inherited
        position = q
        assert spent+incumbent.bound_s <= initial_bound+1e-5
    for point in incumbent.points:
        obs = yield {'kind': 'clear', 'position': point, 'channel': channel,
                     'purpose': 'cover', 'enclosure': poly}
        spent += math.dist(position, point)/5+(5 if obs['kind'] == 'success' else 3)
        position = point
        if obs['kind'] == 'success':
            break
        if obs['kind'] != 'no_target_in_range':
            raise ValueError('unconfirmed clear cannot advance the fallback')
    else:
        raise ValueError('finite cover exhausted without success')
    # Explicitly pay and execute the return. No fictional free teleport/move action.
    obs = yield {'kind': 'measure', 'position': anchor, 'channel': channel, 'purpose': 'return'}
    if obs['kind'] != 'no_signal':
        raise ValueError('cleared channel returned unexpected feedback')
    spent += math.dist(position, anchor)/5+5
    assert spent <= initial_bound+1e-5
    return {'initial_bound_s': initial_bound, 'actual_s': spent, 'decisions': decisions}


def block_channels(current, channels, end):
    middle = sorted(set(channels)-{end})
    if current in middle:
        middle.remove(current)
        middle.insert(0, current)
    return tuple(middle+[end])


def route_cost(route, points, start, channel):
    cost = 0.
    for i, channels in route:
        cost += math.dist(start, points[i])/5
        for c in channels:
            cost += 5+int(c != channel)
            channel = c
        start = points[i]
    return cost


def schedule(required, start=(0., 0.), channel=1, points=POINTS,
             mode='exact', budget_s=math.inf, incumbent_route=None):
    """Anytime: complete nearest plan exists before exact search starts."""
    start_clock = time.perf_counter()
    required = tuple(tuple(sorted(s)) for s in required)
    n = len(points)
    mask = sum(1 << i for i in range(n) if required[i])
    fallback, p, c = [], start, channel
    todo = [i for i in range(n) if required[i]]
    while todo:
        i = todo[0] if mode == 'fixed' else min(todo, key=lambda i: (round(math.dist(p, points[i]), 8), i))
        order = ((c,) + tuple(x for x in required[i] if x != c)) if c in required[i] else required[i]
        fallback.append((i, order))
        todo.remove(i)
        p, c = points[i], order[-1]
    if incumbent_route is not None:
        retained = tuple((i, tuple(c for c in cs if c in required[i])) for i, cs in incumbent_route)
        retained = tuple((i, cs) for i, cs in retained if cs)
        expected = sorted((i, c) for i, cs in enumerate(required) for c in cs)
        if sorted((i, c) for i, cs in retained for c in cs) != expected:
            raise ValueError('incumbent task multiset mismatch')
        if route_cost(retained, points, start, channel) <= route_cost(fallback, points, start, channel):
            fallback = list(retained)
    if mode != 'exact' or budget_s <= 0:
        return tuple(fallback), {'exact': False, 'timed_out': mode == 'exact',
                                 'wall_s': time.perf_counter()-start_clock}
    choices = {}
    locations = tuple(points)+(start,)

    @lru_cache(None)
    def value(mask, last, c):
        if time.perf_counter()-start_clock >= budget_s:
            raise TimeoutError
        if not mask:
            return 0.
        best = (math.inf, 0, 0)
        for i in range(n):
            if not mask & (1 << i):
                continue
            cs = required[i]
            for end in cs:
                switches = len(cs)-1+int(c not in cs or (c == end and len(cs) > 1))
                score = math.dist(locations[last], points[i])/5+5*len(cs)+switches
                candidate = (score+value(mask ^ (1 << i), i, end), i, end)
                if candidate < best:
                    best = candidate
        choices[mask, last, c] = best[1:]
        return best[0]
    try:
        value(mask, n, channel)
    except TimeoutError:
        return tuple(fallback), {'exact': False, 'timed_out': True, 'wall_s': time.perf_counter()-start_clock}
    route, last, c = [], n, channel
    while mask:
        i, end = choices[mask, last, c]
        route.append((i, block_channels(c, required[i], end)))
        mask ^= 1 << i
        last, c = i, end
    assert route_cost(route, points, start, channel) <= route_cost(fallback, points, start, channel)+1e-7
    return tuple(route), {'exact': True, 'timed_out': False, 'wall_s': time.perf_counter()-start_clock}


def controller(mode='exact', group='union', budget_s=math.inf):
    """Q3 only: replan after positive feedback or finishing a measurement block."""
    required = [set(range(1, 21)) for _ in POINTS]
    cleared, excluded = set(), set()
    position, channel = (0., 0.), 1
    cached_route = None
    virtual_s = 0.
    metrics = {'planning': [], 'sources': [], 'responsibility_history': [], 'global_ledger_s': []}
    while any(required):
        route, meta = schedule(required, position, channel, mode=mode, budget_s=budget_s,
                               incumbent_route=cached_route)
        cached_route = list(route)
        ledger = virtual_s+route_cost(route, POINTS, position, channel)+(16-len(cleared))*1746
        assert ledger <= 30569+1e-5
        metrics['global_ledger_s'].append(ledger)
        metrics['planning'].append(meta)
        i, order = route[0]
        discovered = False
        for c in order:
            obs = yield {'kind': 'measure', 'position': POINTS[i], 'channel': c, 'purpose': 'discovery'}
            if obs['kind'] not in ('direction', 'near', 'no_signal'):
                raise ValueError('unconfirmed discovery receipt')
            virtual_s += math.dist(position, POINTS[i])/5+5+int(c != channel)
            position, channel = POINTS[i], c
            required[i].remove(c)
            cached_route[0] = (i, tuple(x for x in cached_route[0][1] if x != c))
            if not cached_route[0][1]:
                cached_route.pop(0)
            if obs['kind'] in ('direction', 'near'):
                for duties in required:
                    duties.discard(c)
                info = yield from localize(position, c, obs, group, budget_s)
                metrics['sources'].append(info)
                virtual_s += info['actual_s']
                cleared.add(c)
                discovered = True
            elif not any(c in duties for duties in required):
                excluded.add(c)
            metrics['responsibility_history'].append(sum(map(len, required)))
            if discovered:
                break
    assert cleared.isdisjoint(excluded) and cleared | excluded == set(range(1, 21))
    return {'completed': True, 'controller_virtual_s': virtual_s, 'cleared_channels': sorted(cleared),
            'excluded_channels': sorted(excluded), **metrics}
