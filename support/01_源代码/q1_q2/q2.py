"""Q2 adaptive second observation using disk-aware conservative sector cells.

Analytical reception tests cover the continuous cell enclosure. Point selection
and worst diameter remain numerical optimizations, not global certificates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

import q1
from q2_geometry import evaluate, prior_cells, reading_interval, reception_margins, reflect_cells


def digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def orient_audit(audit, side):
    result = dict(audit)
    if side == -1:
        result['worst_second_bearing_relative_deg'] *= -1
        result['second_bearing_interval_relative_deg'] = [-v for v in result['second_bearing_interval_relative_deg'][::-1]]
    return result


def validate_config(config, error):
    for key in ('minimum_reception_radius_m', 'maximum_reception_radius_m',
                'target_region_radius_m', 'near_radius_m', 'walking_speed_m_s',
                'geometry_margin_m', 'measurement_time_s'):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError('Require positive finite '+key)
    if not (0 < error < 10 and 2*error < config['minimum_measured_crossing_angle_deg'] < 45
            and config['near_radius_m'] < config['minimum_reception_radius_m']
            <= config['maximum_reception_radius_m']):
        raise ValueError('Unsupported physical or angle configuration')
    for key, minimum in (('prior_angular_cells', 1), ('training_bearing_samples', 3),
                         ('validation_bearing_samples', 3), ('search_population', 5),
                         ('search_iterations', 1)):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < minimum:
            raise ValueError('Invalid integer setting: '+key)
    budget = config.get('maximum_move_distance_m')
    if budget is not None and (not np.isfinite(budget) or budget <= 0):
        raise ValueError('Move budget must be positive and finite')


def candidate(point, cells, error, config):
    q = np.asarray(point, dtype=float)
    if q.shape != (2,) or not np.all(np.isfinite(q)):
        return None
    margin = config['geometry_margin_m']
    budget = config.get('maximum_move_distance_m')
    if budget is not None and np.linalg.norm(q) > budget:
        return None
    # Keep q outside the full first wedge; the upper/lower branch is explicit.
    if q[0] < 0 or q[1] <= q[0]*np.tan(np.deg2rad(error))+margin:
        return None
    slack, nearest = reception_margins(q, cells, config['minimum_reception_radius_m'])
    if slack < margin or nearest <= config['near_radius_m']+margin:
        return None
    try:
        bounds = reading_interval(q, cells, error)
    except ValueError:
        return None
    gamma = np.deg2rad(config['minimum_measured_crossing_angle_deg'])
    if bounds[0] < -np.pi+gamma or bounds[1] > -gamma:
        return None
    return {'interval': bounds, 'reception_margin_m': slack, 'nearest_possible_source_m': nearest}


def choose(first_position, first_bearing_deg, geometry_config, config):
    first = np.asarray(first_position, dtype=float)
    if first.shape != (2,) or not np.all(np.isfinite(first)) or not np.isfinite(first_bearing_deg):
        raise ValueError('Require finite first_position_m[2] and first_bearing_deg')
    error = float(geometry_config['bearing_error_deg'])
    validate_config(config, error)
    cells, rotation = prior_cells(first, first_bearing_deg, error, config)
    R0, L = config['minimum_reception_radius_m'], float(cells[:, 3].max())
    # If one possible source is at r, safe q must satisfy |q|<=r+max(R0,r).
    bound = float(np.min(cells[:, 2]+np.maximum(R0, cells[:, 2])))
    budget = config.get('maximum_move_distance_m')
    if budget is not None:
        bound = min(bound, budget)
    if bound <= config['near_radius_m']:
        raise ValueError('No supported candidate under this movement budget')
    finalists, histories, comparisons = [], [], []
    for side in (1, -1):
        branch_cells = reflect_cells(cells, side)
        evaluated = {}

        def objective(x):
            p = tuple(map(float, x))
            if p in evaluated:
                return evaluated[p][0]
            evidence = candidate(p, branch_cells, error, config)
            if evidence is None:
                return 1e8
            metric = evaluate(p, evidence['interval'], error, config['training_bearing_samples'])
            value = metric['sampled_worst_diameter_m']
            evaluated[p] = (value, evidence)
            return value

        # Include original fixed rule, a refined universal rule, and short-range
        # seeds. Explicit seed candidates remain eligible after optimization.
        for scale in (R0, min(R0, 2*L/3), L, bound*.8, bound*.3):
            for xy in ((.8, .6), (.5, .5), (.6, .7), (.7983022, .6022571)):
                objective(np.asarray(xy)*scale)
        old = np.array([.8*R0, .6*R0])
        old_evidence = candidate(old, branch_cells, error, config)
        if old_evidence:
            comparisons.append({'side': side, 'local_position_m': (old*[1, side]).tolist(),
                                **orient_audit(evaluate(old, old_evidence['interval'], error,
                                                        config['validation_bearing_samples'], True), side)})
        x0 = min(evaluated, key=lambda p: evaluated[p][0]) if evaluated else None
        opt = differential_evolution(objective, [(0., bound), (config['near_radius_m'], bound)],
                                     seed=config['search_seed'], popsize=config['search_population'],
                                     maxiter=config['search_iterations'], tol=1e-7,
                                     polish=False, x0=x0)
        # Dense validation re-ranks finalists, unlike validating only one coarse
        # grid winner. Always retain the original fixed point as a comparison.
        shortlist = sorted(evaluated, key=lambda p: (evaluated[p][0], np.hypot(*p)))[:5]
        if tuple(old) in evaluated and tuple(old) not in shortlist:
            shortlist.append(tuple(old))
        for p in shortlist:
            evidence = evaluated[p][1]
            audit = evaluate(p, evidence['interval'], error, config['validation_bearing_samples'], True)
            finalists.append({'point': p, 'side': side, 'audit': audit, 'evidence': evidence})
        histories.append({'side': side, 'feasible_evaluations': len(evaluated),
                          'optimizer_success': bool(opt.success), 'optimizer_message': str(opt.message),
                          'finalists_validated': len(shortlist)})
    if not finalists:
        raise ValueError('No supported direction-producing candidate; increase move budget or inspect prior')
    selected = min(finalists, key=lambda row: (row['audit']['sampled_worst_diameter_m'],
                                             np.hypot(*row['point']), -row['side']))
    canonical = np.asarray(selected['point'])
    local = canonical*[1, selected['side']]
    actual_reading = selected['side']*selected['audit']['worst_second_bearing_relative_deg']
    # Cross-check selected adverse geometry using the improved general Q1
    # implementation (including its LP and diameter consistency checks).
    general = q1.solve([[0, 0], local], [0, actual_reading], error)
    discrepancy = abs(general['diameter_m']-selected['audit']['sampled_worst_diameter_m'])
    if discrepancy > 1e-5:
        raise ArithmeticError('Q1/Q2 diameter disagreement; inspect numerical geometry')
    world = first+rotation @ local
    audit = orient_audit(selected['audit'], selected['side'])
    return {'status': 'selected', 'strategy': 'disk_aware_sector_enclosure_minimax',
            'first_position_m': first.tolist(), 'first_bearing_deg': float(first_bearing_deg % 360),
            'second_position_m': world.tolist(), 'selected_local_m': local.tolist(),
            'selected_side': selected['side'], 'move_distance_m': float(np.linalg.norm(local)),
            'move_and_measure_time_s': float(np.linalg.norm(local)/config['walking_speed_m_s']+config['measurement_time_s']),
            'maximum_move_distance_m': budget, 'same_channel_switch_time_s': 0.,
            'safe_region_margin_m': selected['evidence']['reception_margin_m'],
            'nearest_possible_source_m': selected['evidence']['nearest_possible_source_m'],
            'prior': {'target_disk_used_to_tighten': True, 'near_radius_used_to_tighten': True,
                      'enclosure': 'union_of_closed_annular_sectors',
                      'cells_rad_rad_m_m': cells.tolist(), 'maximum_range_bound_m': L,
                      'target_region_radius_m': config['target_region_radius_m']},
            'candidate_region': {'definition': 'reception_margins >= margin; nearest > near+margin; separated upper/lower branch; crossing >= gamma; optional move budget',
                                 'minimum_measured_crossing_angle_deg': config['minimum_measured_crossing_angle_deg'],
                                 'is_maximal_feasible_region': False},
            'selection_metric': 'sampled_minimax_of_Q1_angular_polygon_diameter_over_enclosed_readings',
            'audit': audit, 'q1_crosscheck_diameter_m': general['diameter_m'],
            'q1_crosscheck_difference_m': discrepancy, 'search_history': histories,
            'reference_candidates': comparisons, 'global_optimality_claimed': False,
            'formal_interval_certificate': False,
            'reception_guarantee_basis': 'analytic radial endpoint/kink and angular extrema over enclosing cells; explicit floating margin'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    configs = Path(__file__).resolve().parents[1]/'configs'
    parser.add_argument('--config', type=Path, default=configs/'q2.json')
    parser.add_argument('--geometry-config', type=Path, default=configs/'q1.json')
    parser.add_argument('--max-move', type=float, help='Optional movement budget in meters')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8'))
    config = json.loads(args.config.read_text(encoding='utf-8'))
    geometry = json.loads(args.geometry_config.read_text(encoding='utf-8'))
    if args.max_move is not None:
        config['maximum_move_distance_m'] = args.max_move
    result = choose(data['first_position_m'], data['first_bearing_deg'], geometry, config)
    result['effective_config'] = config
    result['input_description'] = data.get('description', 'user-supplied first bearing')
    result['provenance'] = {'hash_scheme': 'sha256_text_lf', 'input_sha256': digest(args.input),
                            'config_sha256': digest(args.config), 'geometry_config_sha256': digest(args.geometry_config),
                            'sources': {name: digest(Path(__file__).with_name(name)) for name in
                                        ('q2.py', 'q2_geometry.py', 'q1.py', 'q1_geometry.py')}}
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding='utf-8')
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
