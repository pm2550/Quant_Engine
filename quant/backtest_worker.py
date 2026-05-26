"""Long-running backtest worker.

Loop:
  1. Wait until other-CPU < threshold (default 20%)
  2. Claim next pending task from DB
  3. Run backtest
  4. Save result
  5. If queue is empty, re-seed and sleep a bit

Run: python -m quant.backtest_worker
Or as systemd service /etc/systemd/system/quant-backtest.service
"""
from __future__ import annotations
import argparse
import logging
import os
import signal
import sys
import time
import traceback
from datetime import datetime

from . import backtest, cpu_throttle, db, task_generator

log = logging.getLogger(__name__)


class Worker:
    def __init__(self, threshold_pct: float, idle_sleep: float, empty_seed: bool):
        self.throttle = cpu_throttle.Throttle(threshold_pct=threshold_pct)
        self.idle_sleep = idle_sleep
        self.empty_seed = empty_seed
        self._running = True
        signal.signal(signal.SIGTERM, self._stop)
        signal.signal(signal.SIGINT, self._stop)
        # Pin to lowest CPU + IO priority so the kernel auto-yields too
        try:
            os.nice(19)
        except Exception:
            pass
        try:
            psutil_p = __import__("psutil").Process(os.getpid())
            psutil_p.ionice(__import__("psutil").IOPRIO_CLASS_IDLE)
        except Exception:
            pass

    def _stop(self, *_):
        log.info("stop requested")
        self._running = False

    def run(self) -> None:
        log.info("worker started pid=%d threshold=%.1f%%", os.getpid(), self.throttle.threshold)
        db.init()
        n_done = 0
        n_fail = 0
        t0 = time.time()
        last_throttle_check = 0.0
        throttle_recheck_after_seconds = 30.0  # only re-sample CPU every 30s

        while self._running:
            # Throttle: only sample CPU periodically to avoid wasting time
            # waiting for samples between fast (0.1s) tasks.
            if time.time() - last_throttle_check > throttle_recheck_after_seconds:
                self.throttle.wait_until_clear()
                last_throttle_check = time.time()
                if not self._running:
                    break

            task = db.claim_next()
            if not task:
                if self.empty_seed:
                    # Try generators in order: refine winners (small, focused),
                    # then walk-forward (large, varied), then base seed
                    added = task_generator.refine_winners(top_n=30, max_neighbors=6)
                    if added == 0:
                        added = task_generator.walk_forward(n_months=18, periods=[3, 5])
                    if added == 0:
                        added = task_generator.seed()
                    log.info("queue empty - reseeded %d new tasks", added)
                    if added == 0:
                        log.info("nothing to add, sleeping %ds", int(self.idle_sleep))
                        time.sleep(self.idle_sleep)
                else:
                    log.info("queue empty, sleeping %ds", int(self.idle_sleep))
                    time.sleep(self.idle_sleep)
                continue

            t_start = time.time()
            try:
                result = backtest.run(
                    task["strategy"], task["symbol"], task["params"],
                    period_years=task["period_years"],
                )
                db.finish(task["id"], result=result)
                n_done += 1
                if n_done % 50 == 0 or n_done <= 3:
                    log.info("done #%d %.2fs %s/%s/%dy params=%s sharpe=%.2f ret=%.1f%%",
                             n_done, time.time() - t_start,
                             task["strategy"], task["symbol"], task["period_years"],
                             task["params"], result["sharpe"], result["total_return"]*100)
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                err = f"{e.__class__.__name__}: {e}"
                db.finish(task["id"], error=err)
                if "insufficient history" not in err:
                    log.warning("FAIL %s/%s params=%s: %s", task["strategy"], task["symbol"], task["params"], err)

        elapsed = time.time() - t0
        log.info("worker stopped: done=%d fail=%d elapsed=%.0fs", n_done, n_fail, elapsed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=20.0,
                    help="if other-CPU%% > this, pause")
    ap.add_argument("--idle-sleep", type=float, default=300.0,
                    help="seconds to sleep when queue is empty")
    ap.add_argument("--no-reseed", action="store_true",
                    help="don't auto re-seed when queue empties")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    Worker(args.threshold, args.idle_sleep, not args.no_reseed).run()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
