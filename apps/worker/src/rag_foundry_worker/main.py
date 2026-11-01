"""Worker entrypoint. The loop does no work yet (job engine is P8.4)."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence

from rag_foundry_core.settings import load_settings

logger = logging.getLogger("rag_foundry_worker")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-foundry-worker")
    parser.add_argument("--once", action="store_true", help="start, load settings and exit")
    args = parser.parse_args(argv)

    settings = load_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s %(message)s")
    logger.info("worker started env=%s", settings.environment)
    if args.once:
        return 0
    try:
        while True:
            time.sleep(settings.worker_poll_seconds)
    except KeyboardInterrupt:
        return 0
