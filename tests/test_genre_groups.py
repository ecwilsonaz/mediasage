"""Tests for smart genre grouping."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend import genre_groups, library_cache
from backend.llm_client import LLMResponse


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(library_cache, "DB_PATH", tmp_path / "cache.db")
    monkeypatch.setattr(library_cache, "DATA_DIR", tmp_path)
    monkeypatch.setattr(library_cache, "_schema_initialized", False)
    monkeypatch.setattr(library_cache, "_cache_from_previous_server", False)


@pytest.fixture(autouse=True)
def enabled():
    with patch.object(genre_groups, "is_enabled", return_value=True) as setting:
        yield setting


def add_tracks(*genre_lists: list[str]) -> None:
    """Replace the cached tracks with one track per genre list."""
    conn = library_cache.ensure_db_initialized()
    conn.execute("DELETE FROM tracks")
    conn.executemany(
        "INSERT INTO tracks (rating_key, title, artist, album, genres, is_live, parent_rating_key) "
        "VALUES (?, 'Song', 'Artist', ?, ?, 0, ?)",
        [(str(i), f"Album {i}", json.dumps(genres), f"a{i}") for i, genres in enumerate(genre_lists)],
    )
    conn.execute("UPDATE sync_state SET track_count = ?", (len(genre_lists),))
    conn.commit()
    conn.close()


class FakeLLM:
    """Answers each grouping call with the next canned reply (a dict, or an exception to raise)."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    def analyze(self, prompt, system, effort=None):
        assert effort == "low"
        self.prompts.append((prompt, system))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(content=json.dumps(reply), input_tokens=100, output_tokens=50, model="test", local=True)

    def parse_json_response(self, response):
        return json.loads(response.content)


def run(llm: FakeLLM | None, full: bool = False):
    with patch.object(genre_groups, "get_llm_client", return_value=llm):
        return genre_groups.update(full)


ROCK_JAZZ = {
    "groups": [
        {"name": "Rock", "description": "Guitar music", "genres": ["Rock", "rock", "Alt Rock"]},
        {"name": "Jazz", "description": "Jazz of all kinds", "genres": ["Jazz"]},
    ],
    "non_genre": ["seen live"],
}


def library() -> None:
    add_tracks(["Rock", "seen live"], ["rock"], ["Alt Rock", "Rock"], ["Jazz"], [])


class TestSplitting:
    def test_splits_on_semicolon_and_comma(self):
        assert library_cache._merge_genres(["Rock; Pop", "Pop , Soul", "rock"]) == ["Rock", "Pop", "Soul"]

    def test_leaves_slashes_and_drops_empties(self):
        assert library_cache._merge_genres(["R&B/Soul", " ; ", ""]) == ["R&B/Soul"]


class TestGenreFilter:
    def test_no_filter(self):
        assert library_cache.genre_filter(None) is None
        assert library_cache.genre_filter([]) is None

    def test_matches_any_tag_case_insensitively(self):
        matches = library_cache.genre_filter(["rock"])
        assert matches(["Jazz", "Rock"]) and not matches(["Jazz"]) and not matches([])

    def test_untagged_is_opt_in(self):
        assert library_cache.genre_filter(["Rock"], include_untagged=True)([])
        only_untagged = library_cache.genre_filter([], include_untagged=True)
        assert only_untagged([]) and not only_untagged(["Rock"])

    def test_filters_use_it(self):
        library()
        assert library_cache.count_tracks_by_filters(genres=["Rock"]) == 3
        assert library_cache.count_tracks_by_filters(genres=["Rock"], include_untagged=True) == 4
        assert len(library_cache.get_tracks_by_filters(genres=["Jazz"], include_untagged=True)) == 2
        assert len(library_cache.get_album_candidates(genres=["Jazz"], include_untagged=True)) == 2
        assert len(library_cache.get_album_candidates(genres=["Jazz"])) == 1


class TestUpdate:
    def test_first_run_groups_every_tag(self):
        library()
        llm = FakeLLM(ROCK_JAZZ)
        status = run(llm)
        assert status["state"] == "ok" and status["tokens"] == 150
        groups, mapping = genre_groups.load()
        assert groups == {"Rock": "Guitar music", "Jazz": "Jazz of all kinds"}
        assert mapping == {"Rock": "Rock", "rock": "Rock", "Alt Rock": "Rock", "Jazz": "Jazz", "seen live": None}
        assert llm.prompts[0][1] == genre_groups.FULL_SYSTEM and "Rock\t2" in llm.prompts[0][0]
        assert genre_groups.is_active()

    def test_unchanged_tags_skip_the_llm(self):
        library()
        run(FakeLLM(ROCK_JAZZ))
        llm = FakeLLM()
        status = run(llm)
        assert status["state"] == "skipped" and "no changes detected" in status["message"] and llm.prompts == []

    def test_new_tags_are_added_to_existing_groups(self):
        add_tracks(["Rock"], ["rock"], ["Alt Rock"], ["Jazz"], ["seen live"], ["Soul"], ["Funk"], ["R&B"], ["Blues"])
        run(FakeLLM({"groups": [
            {"name": "Rock", "description": "Guitar music", "genres": ["Rock", "rock", "Alt Rock"]},
            {"name": "Jazz", "description": "Jazz", "genres": ["Jazz"]},
            {"name": "Soul & Funk", "description": "Soul", "genres": ["Soul", "Funk", "R&B", "Blues"]},
        ], "non_genre": ["seen live"]}))
        add_tracks(["Rock"], ["rock"], ["Alt Rock"], ["Jazz"], ["seen live"], ["Soul"], ["Funk"], ["R&B"], ["Blues"],
                   ["Bebop"], ["Folk"])
        llm = FakeLLM({"groups": [
            {"name": "jazz", "genres": ["Bebop"]},
            {"name": "Folk", "description": "Acoustic traditions", "genres": ["Folk"]},
        ]})
        status = run(llm)
        assert "2 new genre tags grouped" in status["message"]
        prompt, system = llm.prompts[0]
        assert system == genre_groups.ADD_SYSTEM and "- Rock: Guitar music" in prompt
        assert "Bebop\t1" in prompt and "Soul\t" not in prompt  # only the new tags are sent
        groups, mapping = genre_groups.load()
        assert mapping["Bebop"] == "Jazz" and mapping["Folk"] == "Folk" and mapping["Rock"] == "Rock"
        assert groups["Folk"] == "Acoustic traditions"

    def test_removed_tags_and_empty_groups_are_dropped_without_the_llm(self):
        library()
        run(FakeLLM(ROCK_JAZZ))
        add_tracks(["Rock"], ["rock"], ["Alt Rock", "seen live"])
        llm = FakeLLM()
        status = run(llm)
        assert status["state"] == "ok" and "no AI call needed" in status["message"] and llm.prompts == []
        groups, mapping = genre_groups.load()
        assert "Jazz" not in groups and "Jazz" not in mapping

    def test_many_new_tags_regenerate_everything(self):
        library()
        run(FakeLLM(ROCK_JAZZ))
        add_tracks(["Rock"], ["Jazz"], ["Soul"], ["Funk"], ["Folk"])
        llm = FakeLLM({"groups": [{"name": "Everything", "description": "", "genres": ["Rock", "Jazz", "Soul", "Funk", "Folk"]}]})
        status = run(llm)
        assert "fully regenerated" in status["message"]
        assert llm.prompts[0][1] == genre_groups.FULL_SYSTEM
        assert set(genre_groups.load()[1].values()) == {"Everything"}

    def test_full_regroups_even_when_nothing_changed(self):
        library()
        run(FakeLLM(ROCK_JAZZ))
        llm = FakeLLM({"groups": [{"name": "All", "description": "", "genres": ["Rock", "rock", "Alt Rock", "Jazz", "seen live"]}]})
        run(llm, full=True)
        assert set(genre_groups.load()[1].values()) == {"All"}

    def test_skipped_tags_get_a_second_try_then_other(self):
        library()
        llm = FakeLLM(
            {"groups": [{"name": "Rock", "description": "d", "genres": ["Rock", "made up tag"]}]},
            {"groups": [{"name": "Rock", "genres": ["rock"]}], "non_genre": ["seen live"]},
        )
        run(llm)
        _, mapping = genre_groups.load()
        assert mapping == {"Rock": "Rock", "rock": "Rock", "seen live": None, "Alt Rock": "Other", "Jazz": "Other"}
        assert "\nRock\t" not in llm.prompts[1][0].split("New tags")[1]  # the retry only asks about skipped tags

    def test_imperfect_echoes_still_match(self):
        add_tracks(["Chicago Blues\\\\Blues"], ["Hip-Hop"], ["Hip Hop"])
        run(FakeLLM({"groups": [
            {"name": "Blues", "description": "", "genres": ["Chicago Blues\\Blues"]},
            {"name": "Hip-Hop", "description": "", "genres": ["Hip-Hop", "hip hop"]},
        ]}))
        assert genre_groups.load()[1] == {"Chicago Blues\\\\Blues": "Blues", "Hip-Hop": "Hip-Hop", "Hip Hop": "Hip-Hop"}

    def test_case_variants_are_separate_tags(self):
        add_tracks(["Rock"], ["rock"], ["ROCK"])
        run(FakeLLM({"groups": [{"name": "Rock", "description": "", "genres": ["Rock", "rock", "ROCK"]}]}))
        assert genre_groups.load()[1] == {"Rock": "Rock", "rock": "Rock", "ROCK": "Rock"}

    def test_large_libraries_are_grouped_in_batches(self, monkeypatch):
        monkeypatch.setattr(genre_groups, "BATCH_SIZE", 2)
        add_tracks(["A"], ["A"], ["A"], ["B"], ["B"], ["C"])
        llm = FakeLLM(
            {"groups": [{"name": "First", "description": "d", "genres": ["A", "B"]}]},
            {"groups": [{"name": "first", "genres": ["C"]}]},
        )
        run(llm)
        assert set(genre_groups.load()[1].values()) == {"First"}
        assert llm.prompts[0][1] == genre_groups.FULL_SYSTEM and llm.prompts[1][1] == genre_groups.ADD_SYSTEM

    def test_failure_keeps_previous_groups_and_retries_next_time(self):
        library()
        run(FakeLLM(ROCK_JAZZ))
        add_tracks(["Rock"], ["rock"], ["Alt Rock"], ["Jazz"], ["seen live"], ["Bebop"])
        status = run(FakeLLM(RuntimeError("rate limited")))
        assert status["state"] == "error" and "rate limited" in status["message"]
        assert genre_groups.load()[1]["Jazz"] == "Jazz" and genre_groups.is_active()
        assert run(FakeLLM({"groups": [{"name": "Jazz", "genres": ["Bebop"]}]}))["state"] == "ok"

    def test_no_llm(self):
        library()
        assert run(None)["state"] == "error"
        assert not genre_groups.is_active()

    def test_does_nothing_when_off_or_without_tracks(self, enabled):
        assert run(FakeLLM()) is None  # no tracks
        library()
        enabled.return_value = False
        assert run(FakeLLM()) is None

    def test_setting_off_hides_groups_but_keeps_them(self, enabled):
        library()
        run(FakeLLM(ROCK_JAZZ))
        enabled.return_value = False
        assert not genre_groups.is_active()
        assert genre_groups.load()[1]["Jazz"] == "Jazz"


class TestSnapshot:
    def test_groups_with_distinct_track_counts(self):
        library()
        run(FakeLLM(ROCK_JAZZ))
        snapshot = genre_groups.snapshot()
        assert snapshot["groups"] == [
            {"name": "Jazz", "count": 1, "genres": ["Jazz"]},
            {"name": "Rock", "count": 3, "genres": ["Alt Rock", "Rock", "rock"]},  # 3 tracks, 4 tags
        ]
        assert snapshot["non_genre"] == ["seen live"] and snapshot["untagged"] == 1

    def test_tags_not_grouped_yet_show_under_other(self):
        library()
        run(FakeLLM(ROCK_JAZZ))
        add_tracks(["Rock"], ["Bebop"])
        assert genre_groups.snapshot()["groups"][-1] == {"name": "Other", "count": 1, "genres": ["Bebop"]}

    def test_expand(self):
        library()
        run(FakeLLM(ROCK_JAZZ))
        groups = genre_groups.snapshot()["groups"]
        assert genre_groups.expand(["rock", "Nope", None], groups) == ["Alt Rock", "Rock", "rock"]

    def test_estimate(self):
        library()
        estimate = genre_groups.estimate()
        assert estimate["genres"] == 5 and estimate["tokens"] > 0


class TestAfterSyncHook:
    @pytest.fixture(autouse=True)
    def restore_sync_state(self, monkeypatch):
        monkeypatch.setattr(library_cache, "_sync_state", dict(library_cache._sync_state))

    def test_hooks_run_after_a_successful_sync(self, monkeypatch):
        calls = []
        monkeypatch.setattr(library_cache, "after_sync", [lambda: calls.append(library_cache.get_sync_progress()["is_syncing"])])
        client = MagicMock()
        client.get_machine_identifier.return_value = "server"
        client.get_library_identifier.return_value = "lib"
        del client.iter_raw_tracks  # a Jellyfin-style client
        client.get_all_tracks_for_sync.return_value = []
        assert library_cache.sync_library(client)["success"]
        assert calls == [False]  # and the sync is no longer marked as running

    def test_hooks_do_not_run_after_a_failed_sync(self, monkeypatch):
        calls = []
        monkeypatch.setattr(library_cache, "after_sync", [lambda: calls.append(1)])
        client = MagicMock()
        client.get_machine_identifier.return_value = None
        assert not library_cache.sync_library(client)["success"]
        assert calls == []


class TestApi:
    @pytest.fixture
    def client(self):
        from backend.main import app
        return TestClient(app)

    def test_inactive(self, client, enabled):
        enabled.return_value = False
        library()
        data = client.get("/api/genres/groups").json()
        assert data["enabled"] is False and data["active"] is False and data["groups"] == []
        assert data["estimate"]["genres"] == 5

    def test_active(self, client):
        library()
        run(FakeLLM(ROCK_JAZZ))
        data = client.get("/api/genres/groups").json()
        assert data["active"] is True and [g["name"] for g in data["groups"]] == ["Jazz", "Rock"]
        assert data["non_genre"] == ["seen live"] and data["untagged"] == 1 and data["status"]["state"] == "ok"

    def test_regenerate(self, client, enabled):
        with patch("backend.main.get_llm_client", return_value=MagicMock()), \
                patch("backend.main._update_genre_groups") as update:
            assert client.post("/api/genres/groups/regenerate").status_code == 202
            update.assert_called_once_with(full=True)
            enabled.return_value = False
            assert client.post("/api/genres/groups/regenerate").status_code == 400
        enabled.return_value = True
        with patch("backend.main.get_llm_client", return_value=None):
            assert client.post("/api/genres/groups/regenerate").status_code == 503

    def test_turning_the_setting_on_starts_grouping(self, client):
        from tests.test_api import create_mock_config

        config = create_mock_config()
        with patch("backend.main.update_config_values", return_value=config), \
                patch("backend.main.get_config", return_value=config), \
                patch("backend.main.get_current_media_client", return_value=None), \
                patch("backend.main._update_genre_groups") as update:
            assert client.post("/api/config", json={"genre_grouping": True}).status_code == 200
            assert update.call_count == 1
            assert client.post("/api/config", json={"genre_grouping": False}).status_code == 200
            assert update.call_count == 1

    def test_filter_preview_can_include_untagged(self, client):
        library()
        body = {"genres": ["Jazz"], "decades": []}
        assert client.post("/api/filter/preview", json=body).json()["matching_tracks"] == 1
        body["include_untagged"] = True
        assert client.post("/api/filter/preview", json=body).json()["matching_tracks"] == 2

    def test_album_preview_post(self, client):
        library()
        body = {"genres": ["Jazz"], "include_untagged": True}
        assert client.post("/api/recommend/albums/preview", json=body).json()["matching_albums"] == 2


class TestPromptAnalysis:
    def test_ai_picks_groups_which_expand_to_raw_tags(self):
        from backend import analyzer

        library()
        run(FakeLLM(ROCK_JAZZ))
        llm = MagicMock()
        llm.parse_json_response.return_value = {"genres": ["Rock"], "decades": [], "reasoning": ""}
        llm.analyze.return_value = LLMResponse(content="", input_tokens=1, output_tokens=1, model="test", local=True)
        with patch.object(analyzer, "get_llm_client", return_value=llm), \
                patch.object(analyzer, "get_current_media_client", return_value=MagicMock()):
            response = analyzer.analyze_prompt("guitars")
        prompt = llm.analyze.call_args[0][0]
        assert "Jazz (1), Rock (3)" in prompt and "seen live" not in prompt and "Alt Rock" not in prompt
        assert sorted(response.suggested_genres) == ["Alt Rock", "Rock", "rock"]
