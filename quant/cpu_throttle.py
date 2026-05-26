"""CPU usage throttle: pause work when other processes are using CPU."""
from __future__ import annotations
import logging
import os
import time

import psutil

log = logging.getLogger(__name__)


class Throttle:
    """Decide whether to run work this cycle, based on CPU used by *other* processes.

    Strategy: sample CPU over `sample_interval`. Subtract this process's own
    contribution (including children). If the rest exceeds `threshold_pct`,
    sleep for `pause_seconds`. Repeat until the path clears.
    """

    def __init__(
        self,
        threshold_pct: float = 20.0,
        sample_interval: float = 3.0,
        pause_seconds: float = 30.0,
        n_cpus: int | None = None,
    ):
        self.threshold = threshold_pct
        self.sample_interval = sample_interval
        self.pause_seconds = pause_seconds
        self.n_cpus = n_cpus or psutil.cpu_count(logical=True) or 1
        self.me = psutil.Process(os.getpid())
        # Prime the per-process measurement (first call returns 0)
        self.me.cpu_percent(interval=None)

    def measure_other_pct(self, interval: float | None = None) -> float:
        """Return CPU% (0..100) used by everything except this process tree.

        Pass interval=None for an instant non-blocking reading (uses last-sample
        delta), or a positive value to actively sample for that duration.
        """
        # Total CPU% (averaged across cores) over the sample window
        total = psutil.cpu_percent(interval=interval if interval is not None else None)
        # Self CPU% over the same period - psutil returns 0..(100*n_cpus)
        my_pct = self.me.cpu_percent(interval=None) / self.n_cpus
        # Children too
        try:
            for child in self.me.children(recursive=True):
                try:
                    my_pct += child.cpu_percent(interval=None) / self.n_cpus
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except psutil.Error:
            pass
        other = max(0.0, total - my_pct)
        return other

    def wait_until_clear(self, *, sample: bool = True) -> float:
        """Block until other-CPU drops below threshold; return final reading.

        First check uses an active sample (sample_interval seconds) to get a
        reliable reading. If clear, returns immediately. If busy, sleeps and
        re-samples in a loop.
        """
        # Always do at least one active sample to settle the reading
        other = self.measure_other_pct(interval=self.sample_interval if sample else None)
        while other > self.threshold:
            log.info("cpu busy: other=%.1f%% > threshold=%.1f%%, pause %ds",
                     other, self.threshold, int(self.pause_seconds))
            time.sleep(self.pause_seconds)
            other = self.measure_other_pct(interval=self.sample_interval)
        return other
