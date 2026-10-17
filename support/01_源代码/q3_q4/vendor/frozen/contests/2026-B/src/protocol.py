from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = 'http://127.0.0.1:2026'
ARENA_ID = 'default'


class ProtocolError(RuntimeError):
    pass


def _finite_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _valid_text(value, *, limit, field):
    if not isinstance(value, str) or not 1 <= len(value.encode('utf-8')) <= limit:
        raise ValueError(f'{field} must be nonempty UTF-8 text with at most {limit} bytes')
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f'{field} contains control characters')


def _valid_position(x, y):
    for name, value in (('x', x), ('y', y)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'position.{name} must be finite')
        if abs(value) > 2_000_000:
            raise ValueError(f'position.{name} exceeds 2000000 m')


def _valid_channel(channel):
    if isinstance(channel, bool) or not isinstance(channel, int) or not 1 <= channel <= 20:
        raise ValueError('channel must be an integer in 1..20')


@dataclass(frozen=True)
class ProtocolConfig:
    robot_id: str
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = 5.0
    retries: int = 2
    backoff_s: float = 0.05
    user_agent: str = 'cumcm-2026-b/0.1'

    def __post_init__(self):
        _valid_text(self.robot_id, limit=64, field='robot_id')
        if not isinstance(self.base_url, str) or not self.base_url.startswith(('http://127.0.0.1:', 'http://localhost:')):
            raise ValueError('base_url must target the local simulator')
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError('timeout_s must be positive')
        if type(self.retries) is not int or self.retries < 0:
            raise ValueError('retries must be a nonnegative integer')
        if not math.isfinite(self.backoff_s) or self.backoff_s < 0:
            raise ValueError('backoff_s must be nonnegative')


class SimulatorClient:
    """Strict local HTTP client with request-id idempotent retries."""

    def __init__(self, config, opener=None):
        self.config = config
        self.opener = opener or urllib.request.urlopen
        self.session_open = False
        self.virtual_time_s = 0.0
        self.remaining_real_duration_s = None
        self.detector_channel = 1
        self.last_path = None
        self.last_request_id = None
        self.last_payload = None
        self.last_payload_sha256 = None
        self.last_http_status = None
        self._request_counter = 0

    def next_request_id(self, prefix):
        if not isinstance(prefix, str) or not prefix:
            raise ValueError('request prefix must be nonempty')
        self._request_counter += 1
        return f'{prefix}-{self._request_counter}'

    def _request_id(self, prefix):
        return self.next_request_id(prefix)

    def _payload(self, request_id, **fields):
        return {'arena_id': ARENA_ID, 'robot_id': self.config.robot_id,
                'request_id': request_id, **fields}

    def _post(self, path, payload):
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.last_path = path
        self.last_request_id = payload.get('request_id')
        self.last_payload = dict(payload)
        self.last_payload_sha256 = hashlib.sha256(encoded).hexdigest()
        for attempt in range(self.config.retries+1):
            request = urllib.request.Request(
                self.config.base_url+path,
                data=encoded,
                headers={'Content-Type': 'application/json', 'User-Agent': self.config.user_agent},
                method='POST',
            )
            try:
                with self.opener(request, timeout=self.config.timeout_s) as response:
                    status = response.status
                    body = response.read()
            except urllib.error.HTTPError as exc:
                status = exc.code
                body = exc.read()
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt < self.config.retries:
                    time.sleep(self.config.backoff_s*(attempt+1))
                    continue
                raise ProtocolError(f'{path} transport failed: {exc}') from exc
            try:
                reply = json.loads(body.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ProtocolError(f'{path} returned non-JSON response') from exc
            self.last_http_status = status
            if status == 200:
                self._validate_common(reply)
                return reply
            if status in (429, 500, 503) and attempt < self.config.retries:
                time.sleep(self.config.backoff_s*(attempt+1))
                continue
            raise ProtocolError(f'{path} HTTP {status}: {reply}')
        raise ProtocolError('retry loop exhausted')

    @staticmethod
    def _validate_common(reply):
        if not isinstance(reply, dict) or type(reply.get('accepted')) is not bool:
            raise ProtocolError('reply must contain boolean accepted')
        for key in ('real_timestamp_ms', 'virtual_time_s'):
            value = reply.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ProtocolError(f'reply field {key} must be finite')

    @staticmethod
    def _accepted(reply):
        return reply.get('accepted') is True

    def enter(self):
        reply = self._post('/enter', self._payload(self._request_id('enter')))
        if self._accepted(reply):
            for key in ('max_virtual_duration_s', 'max_real_duration_s', 'remaining_real_duration_s'):
                if not _finite_number(reply.get(key)):
                    raise ProtocolError(f'enter reply requires finite {key}')
            self.session_open = True
            self.virtual_time_s = float(reply['virtual_time_s'])
            self.remaining_real_duration_s = float(reply['remaining_real_duration_s'])
        return reply

    def measure_payload(self, request_id, x, y, channel):
        _valid_position(x, y)
        _valid_channel(channel)
        return self._payload(request_id, position={'x': x, 'y': y}, channel=channel)

    def measure(self, x, y, channel, request_id=None):
        if not self.session_open:
            raise ProtocolError('measure requires a successful enter')
        request_id = request_id or self._request_id('measure')
        reply = self._post('/measure', self.measure_payload(request_id, x, y, channel))
        if self._accepted(reply):
            result = reply.get('measure_result')
            if result not in ('no_signal', 'near', 'direction'):
                raise ProtocolError('invalid measure_result')
            if result == 'direction':
                bearing = reply.get('svd_deg')
                if isinstance(bearing, bool) or not isinstance(bearing, (int, float)) or not math.isfinite(bearing):
                    raise ProtocolError('direction reply requires finite svd_deg')
            self.virtual_time_s = float(reply['virtual_time_s'])
            self.detector_channel = channel
        return reply

    def clear_payload(self, request_id, x, y, channel):
        _valid_position(x, y)
        _valid_channel(channel)
        return self._payload(request_id, position={'x': x, 'y': y}, channel=channel)

    def clear(self, x, y, channel, request_id=None):
        if not self.session_open:
            raise ProtocolError('clear requires a successful enter')
        request_id = request_id or self._request_id('clear')
        reply = self._post('/clear', self.clear_payload(request_id, x, y, channel))
        if self._accepted(reply):
            if reply.get('clear_result') not in ('success', 'no_target_in_range'):
                raise ProtocolError('invalid clear_result')
            self.virtual_time_s = float(reply['virtual_time_s'])
        return reply

    def exit(self):
        if not self.session_open:
            raise ProtocolError('exit requires a successful enter')
        reply = self._post('/exit', self._payload(self._request_id('exit')))
        if self._accepted(reply):
            if reply.get('exit_reason') not in ('user_exit',):
                raise ProtocolError('invalid exit_reason')
            self.virtual_time_s = float(reply['virtual_time_s'])
            self.session_open = False
        return reply
