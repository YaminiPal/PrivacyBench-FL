import os
import sys
import logging
from datetime import datetime


def get_logger(name="PrivacyBench-FL", log_dir="logs"):
    """
    Spins up an isolated, dual-stream logging environment for Privacy Benchmarking.
    Prevents handler duplication and ensures zero telemetry data loss.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent logs from bubbling up to Python's root logger (eliminates duplicate outputs)
    logger.propagate = False

    # Your original pattern: Guard against multiple handler registrations upon re-import
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Ensure our target logging directories exist safely on disk
        os.makedirs(log_dir, exist_ok=True)

        # 1. CONSOLE STREAM HANDLER (Immediate stdout tracking)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # 2. TIMESTAMPED FILE HANDLER (Protects history records across separate runs)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(log_dir, f"benchmarking_run_{timestamp}.log")
        
        # delay=True prevents empty file generation if the logger is initiated but unused
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8", delay=True)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Instantiate a clean, framework-wide singleton logger instance
logger = get_logger()