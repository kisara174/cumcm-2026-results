from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction as F
import hashlib
from pathlib import Path
import json
import math
import sys
import time

BASE = Path(__file__).resolve().parents[1]
RESEARCH = BASE/'reviews/r02/b-dispatch-contract/first3/continuation'
REMAINING = RESEARCH/'remaining'
for path in (RESEARCH, REMAINING):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import joint as joint_module
from physical import Belief as ResearchBelief
from refinement import ReviewedController
import certified as cert
from completion import decide_completion
from deadline import planning_limit_s
from ledger import ActionLedger
from results import build_production_record, validate_run_record


def observation_from_reply(path, reply):
    if reply.get('accepted') is not True:
        raise ValueError('cannot convert rejected reply')
    if path == '/measure':
        kind = reply.get('measure_result')
        if kind not in ('no_signal', 'near', 'direction'):
            raise ValueError('invalid measure result')
        observation = {'kind': kind}
        if kind == 'direction':
            observation['bearing'] = reply['svd_deg']
        return observation
    if path == '/clear':
        kind = reply.get('clear_result')
        if kind not in ('success', 'no_target_in_range'):
            raise ValueError('invalid clear result')
        return {'kind': kind}
    raise ValueError('path is not a physical action')


def _positive_polygon_1p01(poly, q, observation):
    q = tuple(map(cert.rat, q))
    if observation['kind'] == 'near':
        return cert.rectangle_clip(poly, q, 5)
    if observation['kind'] != 'direction':
        raise ValueError('positive observation required')
    bearing = F(str(observation['bearing']))
    poly = cert.wedge(poly, q, bearing, half=F(101, 100))
    s, c = cert.trig(bearing)
    poly = cert.interval_halfplane(poly, q, c, s, 1500)
    poly = cert.interval_halfplane(poly, q, cert.neg(c), cert.neg(s), 0)
    transverse = 1500*cert.trig(F(101, 100))[0][1]
    poly = cert.interval_halfplane(poly, q, cert.neg(s), c, transverse)
    poly = cert.interval_halfplane(poly, q, s, cert.neg(c), transverse)
    return cert.rectangle_clip(poly, q, 1500)


class SpecBelief(ResearchBelief):
    """Belief with the Model Spec's closed 1.01 degree report enclosure."""

    report_half = F(101, 100)

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
            self.poly = _positive_polygon_1p01(self.poly, q, obs)
        if kind == 'no_target_in_range':
            self.failed.append(q)
        self.version += 1
        if not self.poly:
            self.inconsistent = True
            raise ValueError('empty certified enclosure')



class AlgorithmController:
    """Spec baseline controller: seven-point Q3, 36-point Q4, complete clear tails."""

    def __init__(self, *, q4=False, use_discovery_dp=True,
                 minimum_sources=10, maximum_sources=16, three_switches=False):
        self.q4 = bool(q4)
        original_belief = joint_module.Belief
        joint_module.Belief = SpecBelief
        try:
            self.base = ReviewedController(
                q4=self.q4,
                joint=three_switches,
                measurements=three_switches,
                use_count=True,
                use_center=False,
                use_discovery_dp=use_discovery_dp,
                use_physical_queries=three_switches,
                minimum_sources=minimum_sources,
                maximum_sources=maximum_sources,
            )
        finally:
            joint_module.Belief = original_belief
        self.plan_id = hashlib.sha256(
            json.dumps(self.base.plan, default=str, sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest()

    def next_action(self):
        raw = self.base.next_action()
        if raw is None:
            return None
        payload = {
            'state_version': len(self.base.history),
            'plan_id': self.plan_id,
            'kind': raw['kind'],
            'channel': raw['channel'],
            'position': [f'{float(v):.6f}' for v in raw['position']],
        }
        action_id = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest()
        return dict(raw, action_id=action_id, state_version=len(self.base.history),
                    plan_id=self.plan_id, branch_id=None)

    def accept_reply(self, path, reply):
        observation = observation_from_reply(path, reply)
        self.base.accept(observation)
        return observation

    def completed(self):
        return self.completion_decision().status == 'completed'

    def completion_decision(self):
        return decide_completion(
            cleared=self.base.cleared,
            excluded=self.base.excluded,
            active_channels=self.base.tasks,
            pending=self.base.pending is not None,
            fault=getattr(self.base, 'fault', None),
            state_version=len(self.base.history),
        )

    def snapshot(self):
        return {
            'position': self.base.position,
            'channel': self.base.channel,
            'spent': self.base.spent,
            'cleared': sorted(self.base.cleared),
            'excluded': sorted(self.base.excluded),
            'tasks': sorted(self.base.tasks),
            'plan_remaining': len(self.base.plan),
            'completion': asdict(self.completion_decision()),
            'completed': self.completed(),
            'optimization_switches': {
                'measurements': self.base.measurements,
                'joint': self.base.joint,
                'use_physical_queries': self.base.use_physical_queries,
                'use_center': self.base.use_center,
                'use_discovery_dp': self.base.use_discovery_dp,
            },
            'optimization_metrics': dict(self.base.metrics),
        }


def run_episode(client, *, q4=False, use_discovery_dp=True, max_actions=10000,
                minimum_sources=10, maximum_sources=16, session_id=None,
                case_id=None, mode='local', spec_path=None, config_paths=(),
                source_paths=(), clock=time.monotonic, three_switches=True):
    # 2026-09-12 起三项优化（额外测向/跨源点级交错/精确物理剪枝）为生产默认；
    # 显式传 three_switches=False 可回退到旧默认用于对照。
    controller = AlgorithmController(
        q4=q4, use_discovery_dp=use_discovery_dp,
        minimum_sources=minimum_sources, maximum_sources=maximum_sources,
        three_switches=three_switches,
    )
    started = clock()
    enter = client.enter()
    if enter.get('accepted') is not True:
        return {
            'schema_version': '2026-B-run/1', 'contest': '2026-B',
            'problem': 4 if q4 else 3, 'command': ['run_episode'],
            'records': [], 'summary': {'completed': False, 'stop_reason': 'enter_rejected'},
        }
    deadline = started+float(enter['remaining_real_duration_s'])
    ledger = ActionLedger(
        session_id or f"{client.config.robot_id}-{4 if q4 else 3}",
        virtual_time_s=float(enter['virtual_time_s']),
    )
    records = []
    stop_reason = 'plan_done'
    while len(records) < max_actions:
        remaining_real = deadline-clock()
        if remaining_real <= 0:
            stop_reason = 'deadline'
            break
        if remaining_real <= 20:
            stop_reason = 'deadline_reserve'
            break
        action = controller.next_action()
        if action is None:
            stop_reason = 'complete' if controller.completed() else 'plan_done'
            break
        path = '/measure' if action['kind'] == 'measure' else '/clear'
        prefix = 'measure' if path == '/measure' else 'clear'
        request_id = client.next_request_id(prefix)
        if path == '/measure':
            payload = client.measure_payload(request_id, *action['position'], action['channel'])
        else:
            payload = client.clear_payload(request_id, *action['position'], action['channel'])
        ledger.begin(path, request_id, payload, action_id=action.get('action_id'))
        try:
            if path == '/measure':
                reply = client.measure(*action['position'], action['channel'], request_id=request_id)
            else:
                reply = client.clear(*action['position'], action['channel'], request_id=request_id)
        except Exception as exc:
            ledger.unknown(str(exc))
            stop_reason = 'undetermined'
            break
        ledger.pending.http_status = client.last_http_status
        if reply.get('accepted') is not True:
            receipt = ledger.reject(path, request_id, payload, reply)
            records.append({
                'request_id': client.last_request_id,
                'action': path,
                'action_id': receipt['action_id'],
                'payload_hash': receipt['payload_hash'],
                'state_version_before': receipt['state_version_before'],
                'state_version_after': receipt['state_version_after'],
                'accepted': False,
                'result_state': receipt['result_state'],
                'virtual_time_s': reply.get('virtual_time_s', 0.),
                'real_timestamp_ms': reply.get('real_timestamp_ms', 0.),
                'response': reply,
            })
            stop_reason = 'rejected'
            break
        receipt = ledger.commit(path, request_id, payload, reply)
        records.append({
            'request_id': client.last_request_id,
            'action': path,
            'action_id': receipt['action_id'],
            'payload_hash': receipt['payload_hash'],
            'position': [payload['position']['x'], payload['position']['y']],
            'channel': payload['channel'],
            'state_version_before': receipt['state_version_before'],
            'state_version_after': receipt['state_version_after'],
            'accepted': True,
            'result_state': receipt['result_state'],
            'virtual_time_s': reply.get('virtual_time_s', 0.),
            'real_timestamp_ms': reply.get('real_timestamp_ms', 0.),
            'planning_limit_s': planning_limit_s(remaining_real),
            'response': reply,
        })
        controller.accept_reply(path, reply)
        if controller.completed():
            stop_reason = 'complete'
            break
    exit_error = None
    if client.session_open:
        try:
            exit_reply = client.exit()
            records.append({
                'request_id': client.last_request_id,
                'action': '/exit',
                'accepted': exit_reply.get('accepted') is True,
                'virtual_time_s': exit_reply.get('virtual_time_s', 0.),
                'real_timestamp_ms': exit_reply.get('real_timestamp_ms', 0.),
                'response': exit_reply,
            })
        except Exception as exc:
            exit_error = repr(exc)
    summary = {
        'completed': controller.completed(),
        'stop_reason': stop_reason,
        'virtual_time_s': client.virtual_time_s,
        'program_wall_s': clock()-started,
        'remaining_real_duration_s': client.remaining_real_duration_s,
        'state': controller.snapshot(),
        'ledger': ledger.snapshot(),
        'exit_error': exit_error,
    }
    if spec_path is None:
        record = {
            'schema_version': '2026-B-run/1',
            'contest': '2026-B',
            'problem': 4 if q4 else 3,
            'command': ['run_episode', 'q4' if q4 else 'q3'],
            'records': records,
            'summary': summary,
        }
        validate_run_record(record)
        return record
    return build_production_record(
        problem=4 if q4 else 3,
        command=['run_episode', 'q4' if q4 else 'q3'],
        records=records,
        summary=summary,
        case_id=case_id or f'q{4 if q4 else 3}-unknown',
        mode=mode,
        spec_path=spec_path,
        config_paths=config_paths,
        source_paths=source_paths,
    )
