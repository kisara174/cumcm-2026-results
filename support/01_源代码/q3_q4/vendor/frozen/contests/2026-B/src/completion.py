from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompletionDecision:
    status: str
    residual_channels: tuple
    evidence: tuple
    state_version: int
    fault: str | None


def decide_completion(*, cleared, excluded, active_channels, pending=False, fault=None,
                      state_version=0, minimum_sources=10, maximum_sources=16,
                      no_active_proof=False):
    cleared = set(map(int, cleared))
    excluded = set(map(int, excluded))
    active = set(map(int, active_channels))
    residual = tuple(sorted(set(range(1, 21))-cleared-excluded))
    if fault:
        return CompletionDecision('undetermined', residual, ('fault',), state_version, str(fault))
    if pending:
        return CompletionDecision('undetermined', residual, ('pending',), state_version, None)
    if not 1 <= minimum_sources <= maximum_sources <= 20:
        return CompletionDecision('undetermined', residual, ('invalid_source_bounds',), state_version, None)
    if not residual:
        return CompletionDecision('completed', (), ('all_channels_cleared_or_excluded',), state_version, None)
    if len(cleared) >= maximum_sources and not active:
        return CompletionDecision('completed', (), ('maximum_source_count_cleared',), state_version, None)
    if no_active_proof and not active:
        return CompletionDecision('completed', (), ('no_active_source_proof',), state_version, None)
    return CompletionDecision('incomplete', residual, ('residual_responsibility',), state_version, None)
