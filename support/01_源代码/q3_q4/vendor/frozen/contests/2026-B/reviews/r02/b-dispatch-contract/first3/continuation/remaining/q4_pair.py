"""Two-observation Q4 visibility certificate (bounded applicability).

If q+ + q- = 2h, h was received, and every possible source is within 1000
of both probes, a no_signal at q+ implies a received signal at q-. Thus the
double-no-signal leaf can be removed with a continuous proof, not sampling.
"""
from fractions import Fraction as F
import math

import certified as cert


def symmetric_pair(poly, anchor, offset):
    anchor = tuple(map(cert.rat, anchor))
    # Command floats are chosen first. Exact symmetry must survive conversion.
    plus = tuple(float(anchor[j]+cert.rat(offset[j])) for j in (0, 1))
    minus = tuple(float(2*anchor[j]-cert.rat(plus[j])) for j in (0, 1))
    exact = all(cert.rat(plus[j])+cert.rat(minus[j]) == 2*anchor[j] for j in (0, 1))
    received_range = all(cert.dist2(p, q) <= 1000**2 for p in poly for q in (plus, minus))
    return {'plus': plus, 'minus': minus, 'certified': exact and received_range,
            'exact_command_symmetry': exact, 'guaranteed_range': received_range,
            'forbidden_feedback_pair': ['no_signal', 'no_signal'] if exact and received_range else None}


def audit():
    # Near the arena's east boundary the first positive's outer polygon is short.
    anchor = (1600., 0.)
    poly = cert.positive_polygon(cert.square(), anchor, {'kind': 'direction', 'bearing': 0})
    good = symmetric_pair(poly, anchor, (0, 100))
    long_poly = cert.positive_polygon(cert.square(), (0, 0), {'kind': 'direction', 'bearing': 0})
    bad = symmetric_pair(long_poly, (0, 0), (0, 100))
    assert good['certified'] and not bad['certified']
    return {'short_prior': good, 'long_prior': bad,
            'scope': 'conditional Q4 two-measurement visibility, not global policy optimality'}
