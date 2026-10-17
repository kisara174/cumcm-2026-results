# 可移植正式入口：默认离线检查，--run 才连接模拟器

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import urllib.request
import urllib.error
import io

sys.dont_write_bytecode = True
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
HERE = Path(__file__).resolve().parent
STUDY = HERE / 'vendor'
ARCHIVE = STUDY / 'frozen-strategy-20260912-heuristics-stop'
POLICY = ARCHIVE / 'selected-policy'
BASE = STUDY / 'frozen/contests/2026-B'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def prepare():
    manifest = json.loads((HERE / 'SOURCE_SHA256.json').read_text(encoding='utf-8'))
    checked = {}
    for rel, expected in manifest.items():
        if sha(STUDY / rel) != expected:
            raise RuntimeError('Frozen source hash mismatch: ' + rel)
        checked[rel] = expected
    sys.path[:0] = [str(POLICY), str(BASE / 'src')]
    import controller
    from policy import solve
    from protocol import ProtocolConfig, SimulatorClient

    assert (
        sha(POLICY / 'policy.py')
        == '4af80777ba07f16c905241de2f11a739a12c153ce34354a5a80b75f1bd003275'
    )
    return (
        checked,
        controller.observation_from_reply,
        solve,
        ProtocolConfig,
        SimulatorClient,
    )


def go_round(value):
    whole, fraction = divmod(abs(value), 1.0)
    rounded = int(whole) + (fraction >= 0.5)
    return -rounded if value < 0 else rounded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--check',
        action='store_true',
        help='Offline source and import validation (default).',
    )
    parser.add_argument(
        '--run',
        action='store_true',
        help='Send live HTTP commands to an already-ready simulator session.',
    )
    parser.add_argument('--output', type=Path)
    parser.add_argument('--problem', type=int, choices=(3, 4))
    parser.add_argument('--case-id')
    parser.add_argument('--mode', choices=('practice', 'formal'), default='practice')
    parser.add_argument('--robot-id')
    args = parser.parse_args()
    checked, observation, solve, Config, Client = prepare()
    binding = dict(
        policy_sha256=sha(POLICY / 'policy.py'),
        source_sha256=checked,
        adapter_sha256=sha(Path(__file__)),
        optimization_resumed=False,
        variants={'3': 'tenth-v1', '4': 'tenth-v1'},
    )
    if args.check or not args.run:
        print(f'CHECK PASS: {len(checked)} frozen source hashes checked; no HTTP calls')
        return
    if not args.output or not args.problem or not args.case_id or not args.robot_id:
        parser.error('--run requires --output, --problem, --case-id, --robot-id')
    if args.robot_id.strip() != args.robot_id:
        parser.error('--robot-id must not contain leading/trailing whitespace')
    out = args.output.resolve()
    if out.exists():
        parser.error(
            '--output must be a new directory; existing records are never overwritten'
        )
    ready = dict(mode=args.mode, problem=args.problem, case_id=args.case_id)
    out.mkdir(parents=True)
    problem = ready['problem']
    variant = binding['variants'][str(problem)]
    save(
        out / 'metadata.json',
        dict(
            binding,
            problem=problem,
            variant=variant,
            case_id=ready['case_id'],
            mode=args.mode,
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ),
    )
    wire = (out / 'wire.jsonl').open('x', encoding='utf-8')
    attempts = (out / 'http-attempts.jsonl').open('x', encoding='utf-8')

    def emit(f, value):
        value = dict(
            value, recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})).open

    def logged_open(request, timeout):
        payload = json.loads(request.data)
        emit(
            attempts,
            dict(
                phase='send',
                path=request.selector,
                payload=dict(payload, robot_id='<redacted>'),
            ),
        )
        try:
            with opener(request, timeout=timeout) as response:
                status = response.status
                body = response.read()
            emit(
                attempts,
                dict(
                    phase='response',
                    request_id=payload['request_id'],
                    status=status,
                    body=body.decode('utf-8', errors='replace'),
                ),
            )
            buffered = io.BytesIO(body)
            buffered.status = status
            return buffered
        except urllib.error.HTTPError as exc:
            body = exc.read()
            emit(
                attempts,
                dict(
                    phase='response',
                    request_id=payload['request_id'],
                    status=exc.code,
                    body=body.decode('utf-8', errors='replace'),
                ),
            )
            raise urllib.error.HTTPError(
                exc.url, exc.code, exc.msg, exc.hdrs, io.BytesIO(body)
            ) from exc
        except Exception as exc:
            emit(
                attempts,
                dict(phase='error', request_id=payload['request_id'], error=repr(exc)),
            )
            raise

    class Recorded(Client):
        def _post(self, path, payload):
            assert path in ('/enter', '/measure', '/clear', '/exit')
            begin = time.perf_counter()
            emit(
                wire,
                dict(
                    phase='send',
                    path=path,
                    payload=dict(payload, robot_id='<redacted>'),
                ),
            )
            try:
                reply = super()._post(path, payload)
            except Exception as exc:
                emit(wire, dict(phase='error', path=path, error=repr(exc)))
                raise
            emit(
                wire,
                dict(
                    phase='response',
                    path=path,
                    reply=reply,
                    http_status=self.last_http_status,
                    elapsed_s=time.perf_counter() - begin,
                ),
            )
            if reply.get('accepted') is not True:
                raise RuntimeError('Simulator rejected action')
            return reply

    client = Recorded(Config(robot_id=args.robot_id), opener=logged_open)
    started = time.perf_counter()
    state = None
    error = None
    exit_error = None
    cleared = set()
    actions = 0
    deadline = None
    max_virtual = None
    try:
        entered = client.enter()
        deadline = time.perf_counter() + entered['remaining_real_duration_s'] - 5
        max_virtual = entered['max_virtual_duration_s']

        def action(kind, p, c):
            nonlocal actions
            if time.perf_counter() >= deadline:
                raise TimeoutError('official remaining real budget')
            if actions >= 10000 or client.virtual_time_s >= max_virtual:
                raise TimeoutError('official action/virtual budget')
            p = tuple(go_round(float(v) * 1e6) / 1e6 for v in p)
            if kind == 'measure':
                reply = client.measure(*p, c)
            elif kind == 'clear':
                reply = client.clear(*p, c)
            else:
                raise ValueError(kind)
            actions += 1
            if kind == 'clear' and reply.get('clear_result') == 'success':
                assert c not in cleared
                cleared.add(c)
            if actions % 100 == 0 or (
                kind == 'clear' and reply.get('clear_result') == 'success'
            ):
                print(
                    json.dumps(
                        dict(
                            problem=problem,
                            actions=actions,
                            cleared=len(cleared),
                            virtual_s=client.virtual_time_s,
                        )
                    ),
                    flush=True,
                )
            return observation('/' + kind, reply)

        state = solve(problem, action, variant)
    except Exception:
        error = traceback.format_exc()
    finally:
        if (
            client.session_open
            and (deadline is None or time.perf_counter() < deadline)
            and (max_virtual is None or client.virtual_time_s < max_virtual)
        ):
            try:
                client.exit()
            except Exception:
                exit_error = traceback.format_exc()
        wire.close()
        attempts.close()
        result = dict(
            problem=problem,
            variant=variant,
            case_id=ready['case_id'],
            mode=args.mode,
            state=state,
            error=error,
            exit_error=exit_error,
            actions=actions,
            cleared_channels=sorted(cleared),
            cleared_count=len(cleared),
            virtual_time_s=client.virtual_time_s,
            tc_s=client.virtual_time_s / len(cleared) if cleared else None,
            wall_s=time.perf_counter() - started,
            session_open=client.session_open,
            total_sources=None,
            full_clearance_verified=False,
        )
        save(out / 'result.json', result)
        print('RESULT', json.dumps(result, ensure_ascii=False), flush=True)
    if error or exit_error or client.session_open:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
