"""Scheduled playlist refresh: regenerate a saved playlist daily or weekly and replace its tracks.

Every run calls the LLM, so it costs what generating the playlist by hand costs. To keep that in check a
failed run waits for its next slot, MAX_FAILURES failures in a row pause the schedule, and runs missed
while the app was down happen once. main.py calls run_due() every minute.
"""

import json
import logging
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator
from zoneinfo import ZoneInfo

from backend import library_cache
from backend.config import get_config, get_current_media_client
from backend.generator import generate_playlist_stream
from backend.llm_client import get_llm_client
from backend.models import GenerateRequest

logger = logging.getLogger(__name__)

MAX_FAILURES = 3
STALE_RUN = timedelta(hours=1)  # A run marked as running for longer than this died with its process
RUNS_PER_MONTH = {"daily": 365 / 12, "weekly": 52 / 12}


class ScheduleError(Exception):
    """A run failed. The message is shown to the user; cost is what the run spent anyway."""

    def __init__(self, message: str, cost: float = 0.0):
        super().__init__(message)
        self.cost = cost


class ScheduleSkipped(Exception):
    """The run doesn't apply right now and isn't a failure."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def next_run_at(frequency: str, hour: int, weekday: int, tz_name: str, after: datetime | None = None) -> datetime:
    """The first slot after `after` (default: now), in UTC. hour is local to tz_name; weekday 0 is Monday."""
    try:
        zone = ZoneInfo(tz_name)
    except Exception:
        zone = ZoneInfo("UTC")
    local = (after or _now()).astimezone(zone)
    slot = local.replace(hour=hour, minute=0, second=0, microsecond=0)
    if frequency == "weekly":
        slot += timedelta(days=(weekday - slot.weekday()) % 7)
    if slot <= local:
        slot += timedelta(days=7 if frequency == "weekly" else 1)
    return slot.astimezone(timezone.utc)


def is_paid_provider() -> bool:
    return get_config().llm.provider not in ("ollama", "custom")


# =============================================================================
# Storage
# =============================================================================


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    conn = library_cache.ensure_db_initialized()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _to_dict(row: sqlite3.Row) -> dict[str, Any]:
    schedule = dict(row)
    schedule["request"] = json.loads(schedule["request"])
    schedule["prompt"] = schedule["request"].get("prompt") or ""
    schedule["enabled"] = bool(schedule["enabled"])
    since = schedule["running_since"]
    schedule["is_running"] = bool(since) and _now() - datetime.fromisoformat(since) < STALE_RUN
    schedule["cost_per_month"] = schedule["cost_per_run"] * RUNS_PER_MONTH[schedule["frequency"]]
    return schedule


def list_schedules() -> list[dict[str, Any]]:
    with _db() as conn:
        rows = conn.execute("SELECT * FROM schedules ORDER BY playlist_title COLLATE NOCASE").fetchall()
    return [_to_dict(row) for row in rows]


def get_schedule(schedule_id: str) -> dict[str, Any] | None:
    with _db() as conn:
        row = conn.execute("SELECT * FROM schedules WHERE id = ?", (schedule_id,)).fetchone()
    return _to_dict(row) if row else None


def create_schedule(
    *,
    playlist_id: str,
    playlist_title: str,
    server_id: str | None,
    request: dict[str, Any],
    frequency: str,
    hour: int,
    weekday: int = 0,
    tz_name: str = "UTC",
    cost_per_run: float = 0.0,
) -> dict[str, Any]:
    """Schedule a playlist, replacing any schedule it already has."""
    schedule_id = secrets.token_hex(8)
    upcoming = next_run_at(frequency, hour, weekday, tz_name).isoformat()
    with _db() as conn:
        conn.execute("DELETE FROM schedules WHERE playlist_id = ? AND server_id IS ?", (playlist_id, server_id))
        conn.execute(
            """INSERT INTO schedules
               (id, playlist_id, playlist_title, server_id, request, frequency, weekday, hour, timezone,
                next_run_at, cost_per_run)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                schedule_id, playlist_id, playlist_title, server_id, json.dumps(request), frequency, weekday,
                hour, tz_name, upcoming, cost_per_run,
            ),
        )
    logger.info("Scheduled %s refresh of playlist '%s'", frequency, playlist_title)
    return get_schedule(schedule_id)


def update_schedule(
    schedule_id: str,
    *,
    enabled: bool | None = None,
    frequency: str | None = None,
    hour: int | None = None,
    weekday: int | None = None,
    tz_name: str | None = None,
) -> dict[str, Any] | None:
    """Pause, resume or retime a schedule; None leaves a value as it is.

    Resuming clears the failure count, and resuming or retiming waits for the next slot.
    """
    schedule = get_schedule(schedule_id)
    if not schedule:
        return None
    timing = {"frequency": frequency, "hour": hour, "weekday": weekday, "timezone": tz_name}
    fields: dict[str, Any] = {column: value for column, value in timing.items() if value is not None}
    if fields or enabled:
        fields["next_run_at"] = _next_slot({**schedule, **fields})
    if enabled is not None:
        fields["enabled"] = int(enabled)
    if enabled:
        fields.update(failures=0, paused_reason=None)
    if fields:
        _write(schedule_id, fields)
    return get_schedule(schedule_id)


def delete_schedule(schedule_id: str) -> bool:
    with _db() as conn:
        return conn.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,)).rowcount > 0


def _next_slot(schedule: dict[str, Any]) -> str:
    return next_run_at(
        schedule["frequency"], schedule["hour"], schedule["weekday"], schedule["timezone"]
    ).isoformat()


def _write(schedule_id: str, fields: dict[str, Any]) -> None:
    assignments = ", ".join(f"{column} = ?" for column in fields)
    with _db() as conn:
        conn.execute(f"UPDATE schedules SET {assignments} WHERE id = ?", [*fields.values(), schedule_id])


def claim(schedule_id: str, only_if_due: bool = False) -> bool:
    """Mark a schedule as running, in one UPDATE so two workers can't both start the same run."""
    now = _now()
    query = "UPDATE schedules SET running_since = ? WHERE id = ? AND (running_since IS NULL OR running_since < ?)"
    params: list[Any] = [now.isoformat(), schedule_id, (now - STALE_RUN).isoformat()]
    if only_if_due:
        query += " AND enabled = 1 AND next_run_at <= ?"
        params.append(now.isoformat())
    with _db() as conn:
        return conn.execute(query, params).rowcount > 0


# =============================================================================
# Running
# =============================================================================


def _generate(request: GenerateRequest) -> tuple[list[str], str, float]:
    """Run a generation to the end. Returns (rating keys, narrative, estimated cost)."""
    rating_keys: list[str] = []
    complete: dict[str, Any] | None = None
    params = request.model_dump(exclude={"seed_track"})
    for chunk in generate_playlist_stream(**params, save_to_history=False):
        # Each chunk is one SSE event: "event: <name>\ndata: <json>\n\n"
        event, _, data = chunk.partition("\ndata: ")
        if not data:
            continue
        event, data = event.removeprefix("event: "), json.loads(data)
        if event == "error":
            raise ScheduleError(data.get("message") or "Generation failed")
        if event == "tracks":
            rating_keys.extend(track["rating_key"] for track in data["batch"])
        elif event == "complete":
            complete = data
    if complete is None or not rating_keys:
        raise ScheduleError("Generation ended without a playlist")
    return rating_keys, complete.get("narrative") or "", complete.get("estimated_cost") or 0.0


def _refresh(schedule: dict[str, Any]) -> tuple[int, float]:
    """Regenerate the playlist and replace its tracks. Returns (track count, cost)."""
    media_client = get_current_media_client()
    if not media_client or not media_client.is_connected():
        raise ScheduleError("Media server not connected")
    if media_client.get_machine_identifier() != schedule["server_id"]:
        raise ScheduleSkipped("This schedule was created on a different media server")
    if not get_llm_client():
        raise ScheduleError("No AI provider is configured")
    # Before generating, so a deleted playlist doesn't cost an LLM call every run
    if not any(p.rating_key == schedule["playlist_id"] for p in media_client.get_playlists()):
        raise ScheduleError("The playlist was not found on the media server. Delete this schedule if it is gone.")

    rating_keys, narrative, cost = _generate(GenerateRequest(**schedule["request"]))
    result = media_client.update_playlist(schedule["playlist_id"], rating_keys, "replace", narrative)
    if not result.get("success"):
        raise ScheduleError(f"Could not update the playlist: {result.get('error') or 'unknown error'}", cost)
    return result.get("tracks_added") or len(rating_keys), cost


def run_schedule(schedule_id: str) -> str:
    """Run a schedule the caller has claimed, and record the outcome: 'ok', 'error' or 'skipped'."""
    schedule = get_schedule(schedule_id)
    title = schedule["playlist_title"]
    fields: dict[str, Any] = {
        "running_since": None,
        "last_run_at": _now().isoformat(),
        "next_run_at": _next_slot(schedule),
        "last_error": None,
    }
    try:
        track_count, cost = _refresh(schedule)
        fields.update(last_status="ok", last_track_count=track_count, failures=0, cost_per_run=cost)
        logger.info("Scheduled refresh of '%s': %d tracks, $%.4f", title, track_count, cost)
    except ScheduleSkipped as e:
        cost = 0.0
        fields.update(last_status="skipped", last_error=str(e))
    except Exception as e:
        if not isinstance(e, ScheduleError):
            logger.exception("Scheduled refresh of '%s' failed", title)
        cost = getattr(e, "cost", 0.0)
        failures = schedule["failures"] + 1
        fields.update(last_status="error", last_error=str(e)[:500], failures=failures)
        if failures >= MAX_FAILURES and schedule["enabled"]:
            fields.update(enabled=0, paused_reason=f"Paused after {failures} failed runs in a row")
        logger.warning("Scheduled refresh of '%s' failed (%d in a row): %s", title, failures, e)
    fields["total_cost"] = schedule["total_cost"] + cost
    _write(schedule_id, fields)
    return fields["last_status"]


def run_due() -> None:
    """Run every enabled schedule whose time has come, one after another."""
    if library_cache.get_sync_progress()["is_syncing"]:
        return  # Wait for the library; due schedules run at the next check
    with _db() as conn:
        due = conn.execute(
            "SELECT id FROM schedules WHERE enabled = 1 AND next_run_at <= ? ORDER BY next_run_at",
            (_now().isoformat(),),
        ).fetchall()
    for row in due:
        if claim(row["id"], only_if_due=True):
            run_schedule(row["id"])
