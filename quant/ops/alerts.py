"""Alerts: push / chat / email notifications for events that need a human.

Every alert is always logged and appended to ``state/alerts.jsonl``. Delivery channels
are optional and configured only through environment variables (``.env``), all free:

    ALERT_NTFY_TOPIC=<long-random-topic>      push to phone via the ntfy app (https://ntfy.sh)
    ALERT_NTFY_SERVER=https://ntfy.sh         (optional; self-hosted server)
    ALERT_WEBHOOK_URL=https://hooks.slack.com/...  or a Discord webhook URL
    ALERT_TELEGRAM_BOT_TOKEN=...  ALERT_TELEGRAM_CHAT_ID=...
    ALERT_SMTP_HOST=smtp.gmail.com ALERT_SMTP_PORT=587 ALERT_SMTP_USER=... ALERT_SMTP_PASSWORD=<app password>
    ALERT_EMAIL_TO=you@example.com            (comma-separated)
    ALERT_MIN_LEVEL=warning                   info | warning | critical

The same ``key`` is sent at most once per ``cooldown`` seconds (default 1 h) so a stuck loop
cannot spam you. Delivery failures are logged and never raise into the trading loop.
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import time
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

logger = logging.getLogger(__name__)
LEVELS = {"info": 10, "warning": 20, "critical": 30}
_EMOJI = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}


class Alerter:
    def __init__(self, state_dir: str | Path = "state", env: dict | None = None, cooldown: float = 3600.0,
                 sender=None):
        self.env = dict(os.environ if env is None else env)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.state_dir / "alerts.jsonl"
        self.cooldown = cooldown
        self.min_level = LEVELS.get(self.env.get("ALERT_MIN_LEVEL", "warning").lower(), 20)
        self._last: dict[str, float] = {}
        self._post = sender  # injectable for tests: callable(url, **kwargs)

    # --- public -----------------------------------------------------------------
    @property
    def channels(self) -> list[str]:
        e = self.env
        ch = []
        if e.get("ALERT_NTFY_TOPIC"):
            ch.append("ntfy")
        if e.get("ALERT_WEBHOOK_URL"):
            ch.append("webhook")
        if e.get("ALERT_TELEGRAM_BOT_TOKEN") and e.get("ALERT_TELEGRAM_CHAT_ID"):
            ch.append("telegram")
        if e.get("ALERT_SMTP_HOST") and e.get("ALERT_EMAIL_TO"):
            ch.append("email")
        return ch

    def send(self, title: str, message: str = "", level: str = "warning", key: str | None = None,
             force: bool = False) -> list[str]:
        """Record and deliver an alert. Returns the channels it was delivered to."""
        level = level if level in LEVELS else "warning"
        key = key or title
        now = time.time()
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "level": level, "key": key, "title": title,
               "message": message}
        getattr(logger, "error" if level == "critical" else level)("ALERT %s: %s %s", level.upper(), title, message)
        if not force and now - self._last.get(key, -1e18) < self.cooldown:
            rec["suppressed"] = True
            self._append(rec)
            return []
        self._last[key] = now
        delivered = []
        if force or LEVELS[level] >= self.min_level:
            for ch in self.channels:
                try:
                    getattr(self, f"_send_{ch}")(title, message, level)
                    delivered.append(ch)
                except Exception as exc:  # never break trading because of an alert
                    logger.warning("alert channel %s failed: %s", ch, exc)
        rec["delivered"] = delivered
        self._append(rec)
        return delivered

    # --- channels ---------------------------------------------------------------
    def _http_post(self, url, **kw):
        if self._post is not None:
            return self._post(url, **kw)
        import requests

        r = requests.post(url, timeout=10, **kw)
        r.raise_for_status()
        return r

    def _send_ntfy(self, title, message, level):
        server = self.env.get("ALERT_NTFY_SERVER", "https://ntfy.sh").rstrip("/")
        prio = {"info": "default", "warning": "high", "critical": "urgent"}[level]
        self._http_post(f"{server}/{self.env['ALERT_NTFY_TOPIC']}", data=(message or title).encode("utf-8"),
                        headers={"Title": title.encode("ascii", "ignore").decode(), "Priority": prio,
                                 "Tags": level})

    def _send_webhook(self, title, message, level):
        text = f"{_EMOJI[level]} *{title}*\n{message}".strip()
        self._http_post(self.env["ALERT_WEBHOOK_URL"], json={"text": text, "content": text})  # Slack | Discord

    def _send_telegram(self, title, message, level):
        tok, chat = self.env["ALERT_TELEGRAM_BOT_TOKEN"], self.env["ALERT_TELEGRAM_CHAT_ID"]
        self._http_post(f"https://api.telegram.org/bot{tok}/sendMessage",
                        json={"chat_id": chat, "text": f"{_EMOJI[level]} {title}\n{message}".strip()})

    def _send_email(self, title, message, level):
        e = self.env
        msg = EmailMessage()
        msg["Subject"] = f"[quant {level.upper()}] {title}"
        msg["From"] = e.get("ALERT_SMTP_USER") or "quant@localhost"
        msg["To"] = e["ALERT_EMAIL_TO"]
        msg.set_content(message or title)
        if self._post is not None:   # test hook
            return self._post("smtp", message=msg)
        with smtplib.SMTP(e["ALERT_SMTP_HOST"], int(e.get("ALERT_SMTP_PORT", 587)), timeout=15) as s:
            s.starttls()
            if e.get("ALERT_SMTP_USER"):
                s.login(e["ALERT_SMTP_USER"], e.get("ALERT_SMTP_PASSWORD", ""))
            s.send_message(msg)

    def _append(self, rec: dict) -> None:
        try:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("cannot write %s: %s", self.log_file, exc)


class NullAlerter(Alerter):
    """Alerter that only logs (used in tests / when state dir is not writable)."""

    def __init__(self):
        self._last = {}
        self.cooldown = 0.0
        self.records: list[dict] = []
        self.env = {}
        self.min_level = 100

    def send(self, title, message="", level="warning", key=None, force=False):
        self.records.append({"title": title, "message": message, "level": level, "key": key or title})
        logger.warning("ALERT %s: %s %s", level.upper(), title, message)
        return []
