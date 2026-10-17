from __future__ import annotations

import multiprocessing as mp
import time


def planning_limit_s(remaining_real_s, reserve_s=20.):
    if remaining_real_s is None:
        return None
    if remaining_real_s <= reserve_s:
        return 0.
    return min(0.2, max(0., remaining_real_s-reserve_s))


def _worker(function, args, queue):
    try:
        queue.put(('ok', function(*args)))
    except Exception as exc:
        queue.put(('error', repr(exc)))


def run_with_hard_deadline(function, args=(), timeout_s=0.2):
    """Run one planning call in a child process and terminate it on timeout."""
    if timeout_s <= 0:
        return {'status': 'skipped_zero_budget', 'result': None, 'wall_s': 0.}
    queue = mp.Queue()
    process = mp.Process(target=_worker, args=(function, args, queue))
    started = time.monotonic()
    process.start()
    process.join(timeout_s)
    wall = time.monotonic()-started
    if process.is_alive():
        process.terminate()
        process.join(1.)
        return {'status': 'timed_out', 'result': None, 'wall_s': wall}
    if queue.empty():
        return {'status': 'worker_failed', 'result': None, 'wall_s': wall}
    status, payload = queue.get()
    return {'status': status, 'result': payload, 'wall_s': wall}
