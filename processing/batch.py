"""Batch processor runs in a background thread and reports progress to the UI."""
import logging
import os
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable, List

from processing.pipeline import (
    ProcessingConfig,
    build_output_path,
    process_image,
    warmup_background_model,
)

logger = logging.getLogger(__name__)


class BatchProcessor:
    def __init__(self):
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()

    def start(
        self,
        image_paths: List[str],
        output_folder: str,
        config: ProcessingConfig,
        on_progress: Callable[[int, int, str], None],
        on_complete: Callable[[int, int], None],
    ):
        """Start batch processing in a background thread."""
        self._cancel_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(image_paths, output_folder, config, on_progress, on_complete),
            daemon=True,
        )
        self._thread.start()

    def cancel(self):
        """Signal the batch to stop after the current in-flight images finish."""
        self._cancel_event.set()

    def _run(self, image_paths, output_folder, config, on_progress, on_complete):
        total = len(image_paths)
        success_count = 0
        fail_count = 0

        try:
            on_progress(0, total, "Loading background model...")
            logger.info("[batch] Warmup started")
            warm_ok = warmup_background_model(
                timeout_sec=config.model_timeout_sec,
                model_name=config.rembg_model,
            )
            if not warm_ok:
                logger.error("[batch] Model warmup failed or timed out")
                on_progress(0, total, "Model load failed. Check logs.")
                on_complete(0, total)
                return
        except Exception as exc:
            logger.exception("[batch] Warmup failed: %s", exc)
            on_progress(0, total, f"Model error: {exc}")
            on_complete(0, total)
            return

        cpu_count = os.cpu_count() or 1
        max_workers = min(total, max(1, min(3, cpu_count // 2 or 1)))
        logger.info("[batch] Using %s worker(s)", max_workers)

        completed = 0
        submitted = 0

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="batch-worker") as executor:
            future_to_path = {}

            def _submit_next() -> bool:
                nonlocal submitted
                if submitted >= total or self._cancel_event.is_set():
                    return False
                input_path = image_paths[submitted]
                output_path = build_output_path(input_path, output_folder, config)
                future = executor.submit(process_image, input_path, output_path, config)
                future_to_path[future] = input_path
                submitted += 1
                return True

            while len(future_to_path) < max_workers and _submit_next():
                pass

            while future_to_path:
                done, _ = wait(tuple(future_to_path.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    input_path = future_to_path.pop(future)
                    try:
                        ok = future.result()
                    except Exception:
                        logger.exception("[batch] Worker crashed while processing %s", input_path)
                        ok = False

                    completed += 1
                    if ok:
                        success_count += 1
                    else:
                        fail_count += 1

                    if self._cancel_event.is_set():
                        on_progress(completed, total, f"Cancelling... {completed}/{total} finished")
                    elif submitted < total:
                        next_name = os.path.basename(image_paths[submitted])
                        on_progress(completed, total, f"{completed}/{total} done | Next: {next_name}")
                    else:
                        on_progress(completed, total, f"{completed}/{total} processed")

                while len(future_to_path) < max_workers and _submit_next():
                    pass

                if self._cancel_event.is_set() and not future_to_path:
                    on_progress(completed, total, f"Cancelled after {completed}/{total}")
                    break

        on_complete(success_count, fail_count)
