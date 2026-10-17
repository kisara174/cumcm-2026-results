"""Certified continuous clear-center B&B and finite information-state minimax.

Never reports global optimality for the full continuous exploration problem.
"""
from fractions import Fraction as F
import heapq
import itertools

import certified as cert


def center_bounds(vertices, box):
    center = ((box[0]+box[1])/2, (box[2]+box[3])/2)
    upper = max(cert.dist2(p, center) for p in vertices)
    lower = max(sum((max(lo-x, F(0), x-hi))**2 for x, lo, hi in
                    ((p[0], box[0], box[1]), (p[1], box[2], box[3]))) for p in vertices)
    return lower, upper, center


def clear_center(vertices, tolerance_m=F(1, 100), max_splits=10000):
    """Global radius optimum for a given rational polygon, with a checked gap."""
    vertices = tuple(tuple(map(cert.rat, p)) for p in vertices)
    if not vertices or tolerance_m <= 0 or max_splits < 0:
        raise ValueError('invalid center optimization inputs')
    box = (min(p[0] for p in vertices), max(p[0] for p in vertices),
           min(p[1] for p in vertices), max(p[1] for p in vertices))
    counter = itertools.count()
    lower, best, point = center_bounds(vertices, box)
    heap = [(lower, next(counter), box)]
    splits = 0
    while heap:
        lower = heap[0][0]
        gap = cert.sqrt_bounds(best)[1]-cert.sqrt_bounds(lower)[0]
        if gap <= tolerance_m or splits >= max_splits:
            break
        _, _, box = heapq.heappop(heap)
        axis = 0 if box[1]-box[0] >= box[3]-box[2] else 2
        mid = (box[axis]+box[axis+1])/2
        for low, high in ((box[axis], mid), (mid, box[axis+1])):
            child = list(box)
            child[axis:axis+2] = [low, high]
            lb, ub, q = center_bounds(vertices, child)
            if ub < best:
                best, point = ub, q
            if lb <= best:
                heapq.heappush(heap, (lb, next(counter), tuple(child)))
        splits += 1
    lower = min(best, heap[0][0]) if heap else best
    command = tuple(map(float, point))
    actual_upper2 = max(cert.dist2(command, p) for p in vertices)
    lo, hi = cert.sqrt_bounds(lower)[0], cert.sqrt_bounds(actual_upper2)[1]
    return {'point': command, 'radius_lower': float(lo), 'radius_upper': float(hi),
            'gap_upper_m': float(hi-lo), 'splits': splits,
            'epsilon_optimal_for_polygon': hi-lo <= tolerance_m,
            'guaranteed_clear_actual_command': actual_upper2 <= 400,
            'scope': 'continuous clear-center for supplied rational polygon only',
            'full_continuous_policy_optimal': False}


def finite_minimax(state, actions, transition, terminal, depth):
    """Information-state recursion, enumerating ALL feedback branches supplied.

    A branch is (observation_label,next_information_state,immediate_cost).
    All functions receive information state, never a selected hidden world.
    Leaf values must represent complete fallback costs for upper-bound meaning.
    """
    best = terminal(state)
    policy = ('fallback',)
    visited = 1
    if depth == 0:
        return best, policy, visited
    for action in actions(state):
        branches = list(transition(state, action))
        if not branches:
            raise ValueError('action lacks complete feedback branches')
        score = F(0)
        children = []
        for label, child, cost in branches:
            value, subpolicy, nodes = finite_minimax(child, actions, transition, terminal, depth-1)
            visited += nodes
            score = max(score, cost+value)
            children.append((label, subpolicy))
        if score < best:
            best, policy = score, (action, tuple(children))
    return best, policy, visited


def depth_witness():
    """Fully enumerated information states: first observation alone is not useful."""
    fallback = {0: F(100), 1: F(100), 2: F(5)}
    def actions(s):
        return ('measure',) if s < 2 else ()
    def transition(s, a):
        return [('left', s+1, F(10)), ('right', s+1, F(10))]
    return [{'depth': d, 'value_s': float(finite_minimax(0, actions, transition, fallback.__getitem__, d)[0])}
            for d in (0, 1, 2, 3)]
