from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import math
from pathlib import Path
import platform
import sys


def sha256_text_lf(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def validate_run_record(record):
    if not isinstance(record, dict):
        raise ValueError('run record must be a mapping')
    required = {'schema_version', 'contest', 'problem', 'command', 'records', 'summary'}
    missing = required-set(record)
    if missing:
        raise ValueError(f'missing run fields: {sorted(missing)}')
    if not isinstance(record['records'], list):
        raise ValueError('records must be a list')
    for index, item in enumerate(record['records']):
        if not isinstance(item, dict):
            raise ValueError(f'records[{index}] must be a mapping')
        for key in ('request_id', 'action', 'accepted', 'virtual_time_s', 'real_timestamp_ms'):
            if key not in item:
                raise ValueError(f'records[{index}] missing {key}')
        if type(item['accepted']) is not bool:
            raise ValueError(f'records[{index}].accepted must be boolean')
        for key in ('virtual_time_s', 'real_timestamp_ms'):
            if not _finite(item[key]):
                raise ValueError(f'records[{index}].{key} must be finite')
    if not isinstance(record['summary'], dict):
        raise ValueError('summary must be a mapping')
    return record


def write_json(path, record):
    validate_run_record(record)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    destination.write_text(text, encoding='utf-8', newline='\n')
    return destination


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def environment_snapshot():
    packages = {}
    for name in ('numpy', 'scipy', 'z3-solver'):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    return {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'packages': packages,
    }


def file_bindings(paths):
    return {Path(path).as_posix(): sha256_text_lf(path) for path in paths}


def build_production_record(*, problem, command, records, summary, case_id, mode,
                            spec_path, config_paths, source_paths):
    if mode not in ('local', 'practice', 'official'):
        raise ValueError('invalid run mode')
    record = {
        'schema_version': '2026-B-production-run/1',
        'contest': '2026-B',
        'problem': int(problem),
        'command': list(command),
        'records': records,
        'summary': summary,
        'case_id': str(case_id),
        'mode': mode,
        'metadata': {
            'environment': environment_snapshot(),
            'spec_sha256': sha256_text_lf(spec_path),
            'config_sha256': file_bindings(config_paths),
            'source_sha256': file_bindings(source_paths),
        },
    }
    validate_run_record(record)
    return record


def validate_production_record(record):
    validate_run_record(record)
    for key in ('case_id', 'mode', 'metadata'):
        if key not in record:
            raise ValueError(f'production record missing {key}')
    if record['mode'] not in ('local', 'practice', 'official'):
        raise ValueError('invalid production mode')
    metadata = record['metadata']
    for key in ('environment', 'spec_sha256', 'config_sha256', 'source_sha256'):
        if key not in metadata:
            raise ValueError(f'production metadata missing {key}')
    return record
