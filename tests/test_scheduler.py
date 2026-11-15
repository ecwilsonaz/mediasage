"""Tests for scheduled playlist refresh."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend import library_cache, scheduler
from backend.models import PlexPlaylistInfo

REQUEST = {
    "prompt": "rainy day jazz",
    "genres": ["Jazz"],
    "decades": [],
    "artists": [],
    "exclude_artists": [],
    "track_count": 2,
    "exclude_live": True,
    "min_rating": 0,
    "max_tracks_to_ai": 500,
}


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(library_cache, "DB_PATH", tmp_path / "cache.db")
    monkeypatch.setattr(library_cache, "DATA_DIR", tmp_path)
    monkeypatch.setattr(library_cache, "_schema_initialized", False)


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def make_schedule(**overrides) -> dict:
    params = dict(
        playlist_id="100", playlist_title="Rainy Jazz", server_id="server-a", request=REQUEST,
        frequency="daily", hour=6, tz_name="UTC", cost_per_run=0.2,
    )
    params.update(overrides)
    return scheduler.create_schedule(**params)


def make_due(schedule_id: str) -> None:
    scheduler._write(schedule_id, {"next_run_at": (scheduler._now() - timedelta(minutes=1)).isoformat()})


def run_now(schedule_id: str) -> str:
    """What the "Run now" endpoint does."""
    return scheduler.run_schedule(schedule_id) if scheduler.claim(schedule_id) else "busy"


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def fake_stream(cost=0.25, keys=("1", "2")):
    def stream(**kwargs):
        stream.calls.append(kwargs)
        yield sse("progress", {"step": "fetching"})
        yield sse("tracks", {"batch": [{"rating_key": k} for k in keys], "index": 0})
        yield sse("complete", {"estimated_cost": cost, "narrative": "A note"})
        yield ": heartbeat\n\n"
    stream.calls = []
    return stream


def failing_stream(**kwargs):
    yield sse("error", {"message": "Rate limit exceeded"})


@pytest.fixture
def media_client():
    client = MagicMock()
    client.is_connected.return_value = True
    client.get_machine_identifier.return_value = "server-a"
    client.get_playlists.return_value = [PlexPlaylistInfo(rating_key="100", title="Rainy Jazz", track_count=2)]
    client.update_playlist.return_value = {"success": True, "tracks_added": 2}
    with patch.object(scheduler, "get_current_media_client", return_value=client), \
            patch.object(scheduler, "get_llm_client", return_value=MagicMock()):
        yield client


class TestNextRun:
    def test_daily_later_today(self):
        assert scheduler.next_run_at("daily", 6, 0, "UTC", utc(2026, 11, 16, 5, 30)) == utc(2026, 11, 16, 6)

    def test_daily_rolls_to_tomorrow_at_the_exact_hour(self):
        assert scheduler.next_run_at("daily", 6, 0, "UTC", utc(2026, 11, 16, 6)) == utc(2026, 11, 17, 6)

    def test_weekly_picks_the_weekday(self):
        # 2026-11-16 is a Monday; weekday 3 = Thursday
        assert scheduler.next_run_at("weekly", 6, 3, "UTC", utc(2026, 11, 16, 12)) == utc(2026, 11, 19, 6)

    def test_weekly_same_day_already_passed(self):
        assert scheduler.next_run_at("weekly", 6, 0, "UTC", utc(2026, 11, 16, 12)) == utc(2026, 11, 23, 6)

    def test_hour_is_local_to_the_timezone(self):
        # Denver is UTC-7 in November
        assert scheduler.next_run_at("daily", 6, 0, "America/Denver", utc(2026, 11, 16, 0)) == utc(2026, 11, 16, 13)

    def test_keeps_local_hour_across_dst_change(self):
        # US clocks go back on 2026-11-01: 06:00 Denver is 12:00 UTC before, 13:00 UTC after
        assert scheduler.next_run_at("daily", 6, 0, "America/Denver", utc(2026, 10, 31, 13)) == utc(2026, 11, 1, 13)

    def test_unknown_timezone_falls_back_to_utc(self):
        assert scheduler.next_run_at("daily", 6, 0, "Not/AZone", utc(2026, 11, 16, 5)) == utc(2026, 11, 16, 6)


class TestStorage:
    def test_create_and_list(self):
        schedule = make_schedule()
        assert schedule["enabled"] and schedule["request"] == REQUEST
        assert schedule["cost_per_run"] == 0.2
        assert schedule["cost_per_month"] == pytest.approx(0.2 * 365 / 12)
        assert [s["id"] for s in scheduler.list_schedules()] == [schedule["id"]]

    def test_weekly_cost_per_month(self):
        assert make_schedule(frequency="weekly")["cost_per_month"] == pytest.approx(0.2 * 52 / 12)

    def test_one_schedule_per_playlist(self):
        first = make_schedule()
        second = make_schedule(frequency="weekly")
        assert [s["id"] for s in scheduler.list_schedules()] == [second["id"]]
        assert scheduler.get_schedule(first["id"]) is None

    def test_same_playlist_id_on_another_server_is_separate(self):
        make_schedule()
        make_schedule(server_id="server-b")
        assert len(scheduler.list_schedules()) == 2

    def test_pause_and_resume_clears_failures(self):
        schedule = make_schedule()
        scheduler._write(schedule["id"], {"failures": 3, "paused_reason": "Paused"})
        assert scheduler.update_schedule(schedule["id"], enabled=False)["enabled"] is False
        resumed = scheduler.update_schedule(schedule["id"], enabled=True)
        assert resumed["enabled"] and resumed["failures"] == 0 and resumed["paused_reason"] is None

    def test_resume_waits_for_the_next_slot(self):
        schedule = make_schedule()
        scheduler.update_schedule(schedule["id"], enabled=False)
        make_due(schedule["id"])
        resumed = scheduler.update_schedule(schedule["id"], enabled=True)
        assert datetime.fromisoformat(resumed["next_run_at"]) > scheduler._now()

    def test_retime(self):
        schedule = make_schedule()
        updated = scheduler.update_schedule(schedule["id"], frequency="weekly", hour=9, weekday=2)
        assert (updated["frequency"], updated["hour"], updated["weekday"]) == ("weekly", 9, 2)
        next_run = datetime.fromisoformat(updated["next_run_at"])
        assert next_run.hour == 9 and next_run.weekday() == 2
        assert updated["cost_per_month"] == pytest.approx(0.2 * 52 / 12)

    def test_retime_keeps_a_paused_schedule_paused(self):
        schedule = make_schedule()
        scheduler._write(schedule["id"], {"enabled": 0, "failures": 3, "paused_reason": "Paused"})
        updated = scheduler.update_schedule(schedule["id"], hour=9)
        assert not updated["enabled"] and updated["failures"] == 3 and updated["hour"] == 9

    def test_pausing_leaves_the_next_slot_alone(self):
        schedule = make_schedule()
        paused = scheduler.update_schedule(schedule["id"], enabled=False)
        assert paused["next_run_at"] == schedule["next_run_at"]

    def test_unknown_schedule(self):
        assert scheduler.update_schedule("0" * 16, enabled=False) is None
        assert scheduler.delete_schedule("0" * 16) is False
        assert scheduler.claim("0" * 16) is False

    def test_delete(self):
        schedule = make_schedule()
        assert scheduler.delete_schedule(schedule["id"]) is True
        assert scheduler.list_schedules() == []


class TestRun:
    def test_due_schedule_replaces_playlist(self, media_client):
        schedule = make_schedule()
        make_due(schedule["id"])
        stream = fake_stream()
        with patch.object(scheduler, "generate_playlist_stream", stream):
            scheduler.run_due()

        media_client.update_playlist.assert_called_once_with("100", ["1", "2"], "replace", "A note")
        assert stream.calls[0]["prompt"] == "rainy day jazz" and stream.calls[0]["genres"] == ["Jazz"]
        assert stream.calls[0]["save_to_history"] is False
        after = scheduler.get_schedule(schedule["id"])
        assert after["last_status"] == "ok" and after["failures"] == 0
        assert after["total_cost"] == 0.25 and after["cost_per_run"] == 0.25
        assert after["last_track_count"] == 2 and not after["is_running"]
        assert datetime.fromisoformat(after["next_run_at"]) > scheduler._now()

    def test_not_due_is_left_alone(self, media_client):
        make_schedule()
        with patch.object(scheduler, "generate_playlist_stream", fake_stream()) as stream:
            scheduler.run_due()
        assert stream.calls == []

    def test_paused_is_not_run_by_the_clock(self, media_client):
        schedule = make_schedule()
        scheduler.update_schedule(schedule["id"], enabled=False)
        make_due(schedule["id"])
        with patch.object(scheduler, "generate_playlist_stream", fake_stream()) as stream:
            scheduler.run_due()
        assert stream.calls == []

    def test_run_now_ignores_the_clock(self, media_client):
        schedule = make_schedule()
        with patch.object(scheduler, "generate_playlist_stream", fake_stream()):
            assert run_now(schedule["id"]) == "ok"

    def test_missed_slots_run_once(self, media_client):
        schedule = make_schedule()
        scheduler._write(schedule["id"], {"next_run_at": (scheduler._now() - timedelta(days=5)).isoformat()})
        with patch.object(scheduler, "generate_playlist_stream", fake_stream()) as stream:
            scheduler.run_due()
            scheduler.run_due()
        assert len(stream.calls) == 1

    def test_failure_waits_for_the_next_slot(self, media_client):
        schedule = make_schedule()
        make_due(schedule["id"])
        with patch.object(scheduler, "generate_playlist_stream", failing_stream):
            scheduler.run_due()
        after = scheduler.get_schedule(schedule["id"])
        assert after["last_status"] == "error" and after["last_error"] == "Rate limit exceeded"
        assert after["failures"] == 1 and after["enabled"]
        assert datetime.fromisoformat(after["next_run_at"]) > scheduler._now()
        media_client.update_playlist.assert_not_called()

    def test_pauses_after_repeated_failures(self, media_client):
        schedule = make_schedule()
        with patch.object(scheduler, "generate_playlist_stream", failing_stream):
            for _ in range(scheduler.MAX_FAILURES):
                make_due(schedule["id"])
                scheduler.run_due()
        after = scheduler.get_schedule(schedule["id"])
        assert not after["enabled"] and after["failures"] == 3
        assert "3 failed runs" in after["paused_reason"]

    def test_success_resets_failure_count(self, media_client):
        schedule = make_schedule()
        scheduler._write(schedule["id"], {"failures": 2})
        with patch.object(scheduler, "generate_playlist_stream", fake_stream()):
            run_now(schedule["id"])
        assert scheduler.get_schedule(schedule["id"])["failures"] == 0

    def test_playlist_update_failure_still_counts_the_cost(self, media_client):
        media_client.update_playlist.return_value = {"success": False, "error": "Playlist not found"}
        schedule = make_schedule()
        with patch.object(scheduler, "generate_playlist_stream", fake_stream(cost=0.3)):
            assert run_now(schedule["id"]) == "error"
        after = scheduler.get_schedule(schedule["id"])
        assert "Playlist not found" in after["last_error"] and after["total_cost"] == 0.3

    def test_deleted_playlist_fails_before_calling_the_llm(self, media_client):
        media_client.get_playlists.return_value = []
        schedule = make_schedule()
        with patch.object(scheduler, "generate_playlist_stream", fake_stream()) as stream:
            assert run_now(schedule["id"]) == "error"
        after = scheduler.get_schedule(schedule["id"])
        assert stream.calls == [] and after["total_cost"] == 0
        assert "not found" in after["last_error"]

    def test_other_media_server_skips_without_failing(self, media_client):
        media_client.get_machine_identifier.return_value = "server-b"
        schedule = make_schedule()
        make_due(schedule["id"])
        with patch.object(scheduler, "generate_playlist_stream", fake_stream()) as stream:
            scheduler.run_due()
        after = scheduler.get_schedule(schedule["id"])
        assert stream.calls == [] and after["last_status"] == "skipped" and after["failures"] == 0
        assert after["enabled"] and datetime.fromisoformat(after["next_run_at"]) > scheduler._now()

    def test_waits_for_library_sync(self, media_client):
        schedule = make_schedule()
        make_due(schedule["id"])
        with patch.object(scheduler, "generate_playlist_stream", fake_stream()) as stream, \
                patch.object(library_cache, "get_sync_progress", return_value={"is_syncing": True}):
            scheduler.run_due()
        after = scheduler.get_schedule(schedule["id"])
        assert stream.calls == [] and after["last_run_at"] is None
        assert datetime.fromisoformat(after["next_run_at"]) < scheduler._now()  # still due

    def test_disconnected_server_is_a_failure(self):
        schedule = make_schedule()
        with patch.object(scheduler, "get_current_media_client", return_value=None):
            assert run_now(schedule["id"]) == "error"
        assert scheduler.get_schedule(schedule["id"])["last_error"] == "Media server not connected"

    def test_running_schedule_is_not_started_twice(self, media_client):
        schedule = make_schedule()
        scheduler._write(schedule["id"], {"running_since": scheduler._now().isoformat()})
        with patch.object(scheduler, "generate_playlist_stream", fake_stream()) as stream:
            assert run_now(schedule["id"]) == "busy"
        assert stream.calls == []

    def test_stale_running_marker_is_ignored(self, media_client):
        schedule = make_schedule()
        scheduler._write(schedule["id"], {"running_since": (scheduler._now() - timedelta(hours=2)).isoformat()})
        with patch.object(scheduler, "generate_playlist_stream", fake_stream()):
            assert run_now(schedule["id"]) == "ok"


class TestScheduleApi:
    @pytest.fixture
    def client(self):
        from backend.main import app
        return TestClient(app)

    @pytest.fixture
    def server(self):
        media = MagicMock()
        media.is_connected.return_value = True
        media.get_machine_identifier.return_value = "server-a"
        media.get_playlists.return_value = [PlexPlaylistInfo(rating_key="100", title="Rainy Jazz", track_count=2)]
        with patch("backend.main.get_current_media_client", return_value=media), \
                patch.object(scheduler, "is_paid_provider", return_value=True):
            yield media

    def save_result(self, result_type="prompt_playlist", request=REQUEST) -> str:
        return library_cache.save_result(
            result_type=result_type, title="Rainy Jazz", prompt="rainy day jazz",
            snapshot={"tracks": [], "estimated_cost": 0.2}, track_count=2, request=request,
        )

    def body(self, result_id, **overrides) -> dict:
        return {
            "result_id": result_id, "playlist_id": "100", "frequency": "daily", "hour": 6,
            "timezone": "America/Denver", "confirm_cost": True, **overrides,
        }

    def test_create_and_list(self, client, server):
        response = client.post("/api/schedules", json=self.body(self.save_result()))
        assert response.status_code == 200, response.text
        created = response.json()
        assert created["playlist_title"] == "Rainy Jazz" and created["prompt"] == "rainy day jazz"
        assert created["cost_per_run"] == 0.2 and created["timezone"] == "America/Denver"

        listing = client.get("/api/schedules").json()
        assert [s["id"] for s in listing["schedules"]] == [created["id"]]
        assert listing["paid_provider"] is True
        assert listing["cost_per_month"] == pytest.approx(0.2 * 365 / 12)

    def test_paid_provider_needs_cost_confirmation(self, client, server):
        response = client.post("/api/schedules", json=self.body(self.save_result(), confirm_cost=False))
        assert response.status_code == 400 and "confirm_cost" in response.json()["detail"]

    def test_local_provider_needs_no_confirmation(self, client, server):
        with patch.object(scheduler, "is_paid_provider", return_value=False):
            response = client.post("/api/schedules", json=self.body(self.save_result(), confirm_cost=False))
        assert response.status_code == 200

    def test_rejects_results_without_parameters(self, client, server):
        # Seed playlists, and anything generated before scheduling existed
        response = client.post("/api/schedules", json=self.body(self.save_result(request=None)))
        assert response.status_code == 400

    def test_unknown_result_and_playlist(self, client, server):
        assert client.post("/api/schedules", json=self.body("0" * 16)).status_code == 404
        response = client.post("/api/schedules", json=self.body(self.save_result(), playlist_id="999"))
        assert response.status_code == 404

    @pytest.mark.parametrize("field,value", [
        ("frequency", "hourly"), ("hour", 24), ("weekday", 7), ("timezone", "Mars/Olympus"),
    ])
    def test_validation(self, client, server, field, value):
        response = client.post("/api/schedules", json=self.body(self.save_result(), **{field: value}))
        assert response.status_code == 422

    def test_pause_delete_and_missing(self, client, server):
        schedule_id = client.post("/api/schedules", json=self.body(self.save_result())).json()["id"]
        paused = client.patch(f"/api/schedules/{schedule_id}", json={"enabled": False})
        assert paused.status_code == 200 and paused.json()["enabled"] is False
        assert client.get("/api/schedules").json()["cost_per_month"] == 0

        edited = client.patch(f"/api/schedules/{schedule_id}", json={"frequency": "weekly", "weekday": 6, "hour": 22})
        assert edited.status_code == 200 and edited.json()["frequency"] == "weekly" and edited.json()["hour"] == 22
        assert edited.json()["enabled"] is False
        assert client.patch(f"/api/schedules/{schedule_id}", json={"timezone": "Mars/Olympus"}).status_code == 422
        assert client.patch(f"/api/schedules/{schedule_id}", json={"hour": 24}).status_code == 422

        assert client.delete(f"/api/schedules/{schedule_id}").status_code == 204
        assert client.delete(f"/api/schedules/{schedule_id}").status_code == 404
        assert client.patch(f"/api/schedules/{schedule_id}", json={"enabled": True}).status_code == 404
        assert client.post(f"/api/schedules/{schedule_id}/run").status_code == 404

    def test_run_now(self, client, server):
        schedule_id = client.post("/api/schedules", json=self.body(self.save_result())).json()["id"]
        with patch.object(scheduler, "run_schedule") as run:
            response = client.post(f"/api/schedules/{schedule_id}/run")
        assert response.status_code == 202 and response.json()["is_running"] is True
        run.assert_called_once_with(schedule_id)
        # Still marked as running (the mocked run never finished), so it can't be started again
        assert client.get("/api/schedules").json()["schedules"][0]["is_running"] is True
        assert client.post(f"/api/schedules/{schedule_id}/run").status_code == 409

    def test_run_now_waits_for_library_sync(self, client, server):
        schedule_id = client.post("/api/schedules", json=self.body(self.save_result())).json()["id"]
        with patch.object(library_cache, "get_sync_progress", return_value={"is_syncing": True}):
            assert client.post(f"/api/schedules/{schedule_id}/run").status_code == 409
        assert client.get("/api/schedules").json()["schedules"][0]["is_running"] is False

    def test_result_reports_schedulable(self, client, server):
        assert client.get(f"/api/results/{self.save_result()}").json()["schedulable"] is True
        assert client.get(f"/api/results/{self.save_result(request=None)}").json()["schedulable"] is False
