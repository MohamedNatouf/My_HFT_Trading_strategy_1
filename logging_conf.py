import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from typing import Optional
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover - fallback not expected on Py>=3.9
    ZoneInfo = None  # type: ignore


class TZFormatter(logging.Formatter):
    """Logging formatter that renders %(asctime)s in a specific timezone.

    Defaults to America/New_York to match US market hours (XNYS). If ZoneInfo
    is unavailable or the timezone can't be resolved, falls back to the
    standard Formatter behavior.
    """

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, tz_name: str = "America/New_York"):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._tz = None
        if ZoneInfo is not None:
            try:
                self._tz = ZoneInfo(tz_name)
            except Exception:
                self._tz = None

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        if self._tz is not None:
            dt = datetime.fromtimestamp(record.created, tz=self._tz)
            if datefmt:
                return dt.strftime(datefmt)
            # Match default logging behavior when no datefmt is provided
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        # Fallback to default implementation
        return super().formatTime(record, datefmt)


def setup_logging(run_id: str = "default", level: int = logging.INFO, tz_name: str = "America/New_York") -> None:
    """Configure root logging to write both to console and per-run file.

    All modules should use `logging.getLogger(__name__)` and will inherit
    this configuration. Logs are written under `logs/<run_id>.log`.
    """

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    log_file = logs_dir / f"{run_id}.log"

    formatter = TZFormatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        tz_name=tz_name,
    )

    root = logging.getLogger()
    # Avoid adding multiple handlers if called more than once
    if getattr(root, "_my_hft_logging_configured", False):
        return

    root.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Rotating file handler (5 MB x 5 backups)
    fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # Mark as configured to avoid duplicates
    root._my_hft_logging_configured = True
