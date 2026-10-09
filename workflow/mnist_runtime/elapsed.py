"""Sleep-inclusive resource time. Performance measurements use a separate clock."""
import math
import sys
import time
from datetime import datetime, timezone
from .common import Blocked

CLOCK_ID = 'darwin-CLOCK_MONOTONIC_RAW/1'


def continuous_seconds():
    if sys.platform != 'darwin' or not hasattr(time, 'CLOCK_MONOTONIC_RAW'):
        raise Blocked('sleep-inclusive resource clock unavailable on this platform')
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) / 1_000_000_000


def observation(boot):
    return {'clock_id': CLOCK_ID, 'continuous_seconds': continuous_seconds(),
            'uptime_seconds': time.monotonic(), 'utc': datetime.now(timezone.utc).isoformat(),
            'boot_identity': boot}


def duration(start, end):
    """No UTC fallback: unknown or incompatible evidence retains the allocation."""
    if not isinstance(start, dict) or not isinstance(end, dict): return None
    if start.get('clock_id') != CLOCK_ID or end.get('clock_id') != CLOCK_ID: return None
    if not start.get('boot_identity') or start['boot_identity'] != end.get('boot_identity'): return None
    values = [start.get('continuous_seconds'), end.get('continuous_seconds')]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values): return None
    elapsed = values[1] - values[0]
    return elapsed if elapsed >= 0 else None


def remaining(origin, current, deadline):
    elapsed = duration(origin, current)
    if elapsed is None: raise Blocked('resource clock identity/boot cannot be reconciled')
    cutoff = datetime.fromisoformat(deadline)
    # The initial UTC-to-continuous binding is frozen. A later backward UTC jump
    # cannot extend it; a forward jump stops admission conservatively.
    initial = (cutoff - datetime.fromisoformat(origin['utc'])).total_seconds()
    utc_left = (cutoff - datetime.fromisoformat(current['utc'])).total_seconds()
    return min(initial - elapsed, utc_left)
