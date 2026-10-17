"""Small trusted arithmetic kernel: rational enclosures, not epsilon predicates.

Certificates apply to the explicitly represented coordinates and closed bearing
error enclosure. This is executable checking, not a proof-assistant artifact.
"""
from fractions import Fraction as F
from functools import lru_cache
from math import factorial, isqrt

SCALE = 10**15


def rat(x):
    return x if isinstance(x, F) else F(x)


def outward(a, b):
    return F(a*SCALE//1, SCALE), F(-((-b*SCALE)//1), SCALE)


def add(a, b):
    return outward(a[0]+b[0], a[1]+b[1])


def mul(a, b):
    p = [x*y for x in a for y in b]
    return outward(min(p), max(p))


def neg(a):
    return -a[1], -a[0]


@lru_cache(None)
def pi_interval():
    def atan(n):
        value = sum((F((-1)**k, (2*k+1)*n**(2*k+1)) for k in range(32)), F(0))
        tail = F(1, 65*n**65)
        return value, value+tail  # 32-term alternating sum ends negative.
    a, b = atan(5), atan(239)
    return 16*a[0]-4*b[1], 16*a[1]-4*b[0]


@lru_cache(maxsize=8192)
def trig(degrees):
    """sin/cos of an exact rational degree value, rigorous Taylor remainder."""
    degree = (rat(degrees)+180) % 360-180
    x = mul((degree/180, degree/180), pi_interval())
    x2 = mul(x, x)
    sin, cos = (F(0), F(0)), (F(0), F(0))
    power = (F(1), F(1))
    for k in range(19):
        cc = F((-1)**k, factorial(2*k))
        ss = F((-1)**k, factorial(2*k+1))
        cos = add(cos, mul(power, (cc, cc)))
        sin = add(sin, mul(mul(power, x), (ss, ss)))
        power = mul(power, x2)
    # |x|<4; use a shared overestimate of degree-36/37 Taylor remainders.
    error = F(4**37, factorial(37))
    return ((max(F(-1), sin[0]-error), min(F(1), sin[1]+error)),
            (max(F(-1), cos[0]-error), min(F(1), cos[1]+error)))


def sqrt_bounds(x, scale=10**9):
    x = rat(x)
    if x < 0:
        raise ValueError('negative squared distance')
    k = isqrt(x.numerator*scale*scale//x.denominator)
    lo = F(k, scale)
    hi = lo if lo*lo == x else F(k+1, scale)
    return lo, hi


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def dist2(a, b):
    return sum((rat(x)-rat(y))**2 for x, y in zip(a, b))


def clip(poly, n, offset):
    if not poly:
        return ()
    out = []
    a = poly[-1]
    fa = dot(n, a)-offset
    for b in poly:
        fb = dot(n, b)-offset
        if (fa <= 0) != (fb <= 0):
            t = fa/(fa-fb)
            out.append(tuple(a[j]+t*(b[j]-a[j]) for j in (0, 1)))
        if fb <= 0:
            out.append(b)
        a, fa = b, fb
    return tuple(out)


def square(q=(0, 0), r=1800):
    x, y = map(rat, q)
    r = rat(r)
    return ((x-r, y-r), (x+r, y-r), (x+r, y+r), (x-r, y+r))


def rectangle_clip(poly, q, r):
    for n in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        poly = clip(poly, n, dot(n, q)+r)
    return poly


def interval_halfplane(poly, q, nx, ny, bound=0):
    """All x in arena satisfying true n.(x-q)<=bound survive midpoint clip."""
    n = ((nx[0]+nx[1])/2, (ny[0]+ny[1])/2)
    slack = (nx[1]-nx[0])/2*(1800+abs(q[0]))+(ny[1]-ny[0])/2*(1800+abs(q[1]))
    return clip(poly, n, dot(n, q)+bound+slack)


def wedge(poly, q, bearing, half=F(201, 200)):
    lower_s, lower_c = trig(rat(bearing)-half)
    upper_s, upper_c = trig(rat(bearing)+half)
    poly = interval_halfplane(poly, q, lower_s, neg(lower_c))
    return interval_halfplane(poly, q, neg(upper_s), upper_c)


def positive_polygon(poly, q, obs):
    q = tuple(map(rat, q))
    if obs['kind'] == 'near':
        return rectangle_clip(poly, q, 5)
    if obs['kind'] != 'direction':
        raise ValueError('positive observation required')
    bearing = F(str(obs['bearing']))
    poly = wedge(poly, q, bearing)
    s, c = trig(bearing)
    poly = interval_halfplane(poly, q, c, s, 1500)
    poly = interval_halfplane(poly, q, neg(c), neg(s), 0)
    transverse = 1500*trig(F(201, 200))[0][1]
    poly = interval_halfplane(poly, q, neg(s), c, transverse)
    poly = interval_halfplane(poly, q, s, neg(c), transverse)
    return rectangle_clip(poly, q, 1500)


def certified_cover(poly, bearing=0, failed=(), allow_empty=False):
    """Whole transformed rectangle tiled; failed disks discard only full cells.

Returns actual binary float commands and exact parallelogram cells. Cover check
uses those actual floats as exact rationals, not untransmitted ideal centers.
"""
    if not poly:
        raise ValueError('empty geometry is inconsistency, never completion')
    s, c = trig(F(str(bearing)))
    sn, cs = sum(s)/2, sum(c)/2
    u, v = (cs, sn), (-sn, cs)
    scale = cs*cs+sn*sn
    xs, ys = [dot(u, p) for p in poly], [dot(v, p) for p in poly]
    lo, hi, bot, top = min(xs), max(xs), min(ys), max(ys)
    nx, ny = max(1, -(-(hi-lo)//25)), max(1, -(-(top-bot)//25))
    def world(x, y):
        return ((cs*x-sn*y)/scale, (sn*x+cs*y)/scale)
    cells, skipped = [], 0
    for i in range(nx):
        for j in (range(ny) if i % 2 == 0 else range(ny-1, -1, -1)):
            xl, xh = lo+i*(hi-lo)/nx, lo+(i+1)*(hi-lo)/nx
            yl, yh = bot+j*(top-bot)/ny, bot+(j+1)*(top-bot)/ny
            vertices = tuple(world(x, y) for x, y in ((xl, yl), (xh, yl), (xh, yh), (xl, yh)))
            if any(all(dist2(z, q) <= 400 for z in vertices) for q in failed):
                skipped += 1
                continue
            center = tuple(float(x) for x in world((xl+xh)/2, (yl+yh)/2))
            assert all(dist2(z, center) <= 400 for z in vertices)
            cells.append((center, vertices))
    if not cells and not allow_empty:
        raise ValueError('all cells excluded: inconsistent observations')
    return cells, {'cells': len(cells), 'excluded_cells': skipped,
                   'exact_rational_cell_coverage': True,
                   'report_enclosure_model': 'closed_1p01',
                   'report_enclosure_half_deg': 1.01}


def q4_grid():
    return tuple((float(x), float(y)) for i, x in enumerate(range(-1800, 1801, 720))
                 for y in (range(-1800, 1801, 720) if i % 2 == 0 else range(1800, -1801, -720)))


def verify_q4_grid():
    """Reflected source lies in a corner triangle; one corner is in every closed halfplane."""
    points = set(q4_grid())
    assert len(points) == 36
    assert (3600//720)**2 == 25
    for x in range(-1800, 1800, 720):
        for y in range(-1800, 1800, 720):
            assert {(x, y), (x+720, y), (x, y+720), (x+720, y+720)} <= points
    assert 5*720**2 < 4*1000**2
    assert 5*900**2 > 4*1000**2
    return {'points': 36, 'max_required_distance_squared_m2': 648000,
            'receive_radius_squared_m2': 1000000,
            'proof': 'reflected source is in the right triangle conv{(0,0),(s,0),(0,s)}; '
                     'every closed halfplane through it holds a cell corner within 5s^2/4',
            'exact_integer_checks': True}
