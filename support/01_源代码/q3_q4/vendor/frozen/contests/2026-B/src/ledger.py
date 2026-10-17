from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time


def payload_hash(payload):
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class PendingAction:
    session_id: str
    request_id: str
    path: str
    payload_hash: str
    payload: dict
    action_id: str | None
    sent_monotonic_s: float
    attempts: int = 1
    http_status: int | None = None
    accepted: bool | None = None


class ActionLedger:
    """Single-writer action ledger with idempotent pending ownership."""

    def __init__(self, session_id, *, position=(0., 0.), channel=1, virtual_time_s=0.):
        self.session_id = str(session_id)
        self.position = tuple(map(float, position))
        self.channel = int(channel)
        self.virtual_time_s = float(virtual_time_s)
        self.cleared = set()
        self.excluded = set()
        self.history = []
        self.state_version = 0
        self.pending = None
        self.fault = None
        self.timing_anomaly = False
        self.max_timing_delta_s = 0.

    def begin(self, path, request_id, payload, *, action_id=None, sent_monotonic_s=None):
        if self.fault:
            raise ValueError(f'ledger fault: {self.fault}')
        digest = payload_hash(payload)
        if self.pending is not None:
            same = (self.pending.path == path and
                    self.pending.request_id == request_id and
                    self.pending.payload_hash == digest)
            if not same:
                raise ValueError('another action is pending')
            self.pending.attempts += 1
            return self.pending
        if path not in ('/measure', '/clear'):
            raise ValueError('ledger only owns physical measure/clear actions')
        if payload.get('request_id') != request_id:
            raise ValueError('request_id does not match payload')
        self.pending = PendingAction(
            session_id=self.session_id,
            request_id=request_id,
            path=path,
            payload_hash=digest,
            payload=dict(payload),
            action_id=action_id,
            sent_monotonic_s=time.monotonic() if sent_monotonic_s is None else float(sent_monotonic_s),
        )
        return self.pending

    def _check_pending(self, path, request_id, payload):
        if self.pending is None:
            raise ValueError('no pending action')
        digest = payload_hash(payload)
        if (self.pending.path != path or self.pending.request_id != request_id or
                self.pending.payload_hash != digest):
            raise ValueError('reply does not match pending action')

    def _expected_cost(self, path, payload, result_state):
        position = (float(payload['position']['x']), float(payload['position']['y']))
        move = math.dist(self.position, position)/5
        if path == '/measure':
            return move+5+int(self.channel != payload['channel']), position, payload['channel']
        if result_state == 'success':
            return move+5, position, self.channel
        return move+3, position, self.channel

    def commit(self, path, request_id, payload, reply, *, received_monotonic_s=None):
        if reply.get('accepted') is not True:
            raise ValueError('commit requires accepted=true')
        self._check_pending(path, request_id, payload)
        self.pending.accepted = True
        if path == '/measure':
            result_state = reply.get('measure_result')
            if result_state not in ('no_signal', 'near', 'direction'):
                raise ValueError('invalid measure_result')
        else:
            result_state = reply.get('clear_result')
            if result_state not in ('success', 'no_target_in_range'):
                raise ValueError('invalid clear_result')
        expected, position, channel = self._expected_cost(path, payload, result_state)
        server_delta = float(reply['virtual_time_s'])-self.virtual_time_s
        tolerance = 2*(len(self.history)+2)*10**-6
        delta = abs(server_delta-expected)
        self.max_timing_delta_s = max(self.max_timing_delta_s, delta)
        if delta > tolerance:
            self.timing_anomaly = True
        self.position = position
        self.channel = channel
        self.virtual_time_s = float(reply['virtual_time_s'])
        if path == '/clear' and result_state == 'success':
            self.cleared.add(payload['channel'])
        self.state_version += 1
        receipt = {
            'session_id': self.session_id,
            'request_id': request_id,
            'payload_hash': self.pending.payload_hash,
            'action_id': self.pending.action_id,
            'http_status': self.pending.http_status,
            'accepted': True,
            'result_state': result_state,
            'raw_response': dict(reply),
            'sent_monotonic_s': self.pending.sent_monotonic_s,
            'received_monotonic_s': time.monotonic() if received_monotonic_s is None else float(received_monotonic_s),
            'state_version_before': self.state_version-1,
            'state_version_after': self.state_version,
            'expected_cost_s': expected,
            'server_cost_s': server_delta,
            'timing_delta_s': delta,
        }
        self.history.append(receipt)
        self.pending = None
        return receipt

    def reject(self, path, request_id, payload, reply, *, received_monotonic_s=None):
        self._check_pending(path, request_id, payload)
        self.pending.accepted = False
        if reply.get('accepted') is not False:
            raise ValueError('reject requires accepted=false')
        receipt = {
            'session_id': self.session_id,
            'request_id': request_id,
            'payload_hash': self.pending.payload_hash,
            'action_id': self.pending.action_id,
            'http_status': self.pending.http_status,
            'accepted': False,
            'result_state': 'rejected',
            'raw_response': dict(reply),
            'sent_monotonic_s': self.pending.sent_monotonic_s,
            'received_monotonic_s': time.monotonic() if received_monotonic_s is None else float(received_monotonic_s),
            'state_version_before': self.state_version,
            'state_version_after': self.state_version,
        }
        self.pending = None
        self.history.append(receipt)
        return receipt

    def unknown(self, reason):
        if self.pending is None:
            raise ValueError('no pending action')
        self.pending.attempts += 1
        return {
            'state_version': self.state_version,
            'request_id': self.pending.request_id,
            'payload_hash': self.pending.payload_hash,
            'reason': str(reason),
            'pending': True,
        }

    def pending_json(self):
        if self.pending is None:
            return None
        return json.dumps(asdict(self.pending), ensure_ascii=False, sort_keys=True)

    def snapshot(self):
        return {
            'session_id': self.session_id,
            'state_version': self.state_version,
            'position_m': self.position,
            'channel': self.channel,
            'virtual_time_s': self.virtual_time_s,
            'cleared_channels': sorted(self.cleared),
            'excluded_channels': sorted(self.excluded),
            'pending': None if self.pending is None else asdict(self.pending),
            'fault': self.fault,
            'timing_anomaly': self.timing_anomaly,
            'max_timing_delta_s': self.max_timing_delta_s,
        }
