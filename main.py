#!/usr/bin/env python3
"""ClothingSnap entry point."""
import logging
import os
import sys
import time

# Ensure app directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main() -> int:
    startup = time.perf_counter()
    logger.info("[main] Starting ClothingSnap")

    logger.info("[main] Importing UI module")
    from ui.app_window import ClothingSnapApp

    logger.info("[main] Building application window")
    app = ClothingSnapApp()

    logger.info("[main] Entering Tk mainloop (startup %.2fs)", time.perf_counter() - startup)
    app.run()
    logger.info("[main] Tk mainloop exited")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("[main] Fatal startup error")
        raise
