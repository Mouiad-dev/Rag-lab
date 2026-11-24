"""Worker entrypoint. The loop does no work yet (job engine is P8.4)."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from rag_foundry_core.settings import load_settings

logger = logging.getLogger("rag_foundry_worker")


def is_fresh(path: Path, max_age: float, now: float) -> bool:
    """Return True if path exists and its mtime is at most max_age seconds old (not in the future)."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return False
    return 0 <= now - mtime <= max_age


def _touch(path: Path | None) -> None:
    if path is not None:
        path.touch()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-foundry-worker")
    parser.add_argument("--once", action="store_true", help="start, load settings and exit")
    parser.add_argument(
        "--heartbeat-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="touch this file at start and on every loop iteration",
    )
    parser.add_argument(
        "--check-heartbeat",
        type=Path,
        default=None,
        metavar="PATH",
        help="exit 0 if PATH is fresh, 1 otherwise; does not start the worker",
    )
    args = parser.parse_args(argv)
    if args.check_heartbeat is not None and (args.once or args.heartbeat_file is not None):
        parser.error("--check-heartbeat cannot be combined with --once or --heartbeat-file")

    settings = load_settings()

    if args.check_heartbeat is not None:
        max_age = 3 * settings.worker_poll_seconds + 5
        return 0 if is_fresh(args.check_heartbeat, max_age, time.time()) else 1

    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s %(message)s")
    logger.info("worker started env=%s", settings.environment)
    _touch(args.heartbeat_file)
    if args.once:
        return 0
    try:
        while True:
            time.sleep(settings.worker_poll_seconds)
            _touch(args.heartbeat_file)
    except KeyboardInterrupt:
        return 0
