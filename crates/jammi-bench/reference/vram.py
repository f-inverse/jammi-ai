"""`jammi_bench::vram`, the one device-memory instrument every rung is
measured with: the whole-device `memory.used` figure `nvidia-smi` reports,
polled every 25 ms on a background thread, reduced to the high-water mark
above a baseline read when the window opens. A driver-level pool figure,
never the framework's own allocator counters — those are provenance beside
it, not the compared column. Shared by every twin that measures a device.
"""
from __future__ import annotations

import subprocess
import threading

POLL_INTERVAL_S = 0.025


def nvidia_smi_memory_used(ordinal=None):
    """`vram::nvidia_smi_memory_used`: the first line of the `memory.used`
    query for device `ordinal` (the whole box's first device when `None`),
    MiB → bytes; `None` when the host cannot say."""
    command = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    if ordinal is not None:
        command.insert(1, f"--id={ordinal}")
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        if out.returncode != 0:
            return None
        return int(out.stdout.splitlines()[0].strip()) * 1024 * 1024
    except (OSError, subprocess.TimeoutExpired, IndexError, ValueError):
        return None


class VramWindow:
    """`vram::VramWindow`: a baseline read at construction, then a background
    poll of the same probe every `POLL_INTERVAL_S`; `close()` is the
    high-water mark above the baseline in bytes, or `None` when nothing
    could be sampled. Open it where the engine opens its own — after the
    model, adapters and optimizer are resident, before any step — so both
    windows start from the same pool state."""

    def __init__(self, ordinal=None, probe=nvidia_smi_memory_used):
        self._probe = lambda: probe(ordinal)
        first = self._probe()
        self._baseline = first or 0
        self._peak = 0
        self._stop = threading.Event()
        self._thread = None
        if first is not None:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()

    def _poll(self):
        while not self._stop.is_set():
            used = self._probe()
            if used is not None:
                self._peak = max(self._peak, used)
            self._stop.wait(POLL_INTERVAL_S)

    def close(self):
        if self._thread is None:
            return None
        self._stop.set()
        self._thread.join()
        return float(max(self._peak - self._baseline, 0))
