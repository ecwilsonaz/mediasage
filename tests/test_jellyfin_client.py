"""Tests for the Jellyfin client, against a fake Jellyfin API."""

import json

import httpx
import pytest

from backend import jellyfin_client, library_cache
from backend.jellyfin_client import SCRATCH_PLAYLIST_TITLE, JellyfinClient

USER_ID = "u" * 32
LIBRARY_ID = "1" * 32
ALBUM_ID = "a" * 32


def _audio(n: int, **extra) -> dict:
    item = {
        "Id": f"{n:032x}",
        "Name": f"Track {n}",
        "AlbumArtist": "Artist",
        "Album": "Album",
        "AlbumId": ALBUM_ID,
        "AlbumPrimaryImageTag": "tag",
        "RunTimeTicks": 2_000_000_000,
        "ProductionYear": 1994,
        "Genres": ["Jazz"],
        "ImageTags": {},
        "UserData": {"PlayCount": n, "LastPlayedDate": "2026-01-02T03:04:05Z"},
    }
    item.update(extra)
    return item


class FakeJellyfin:
    """Just enough of the Jellyfin API for JellyfinClient."""

    def __init__(self, tracks: list[dict]):
        self.tracks = tracks
        self.playlists: dict[str, dict] = {}
        self.fail_paths: set[str] = set()
        self.fail_next_add = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = request.url.params
        if path in self.fail_paths:
            return httpx.Response(500)

        if path == "/Users":
            return httpx.Response(200, json=[{"Id": USER_ID, "Policy": {"IsAdministrator": True}}])
        if path == "/Library/MediaFolders":
            return httpx.Response(200, json={"Items": [
                {"Id": "m" * 32, "Name": "Movies", "CollectionType": "movies"},
                {"Id": LIBRARY_ID, "Name": "Music", "CollectionType": "music"},
            ]})
        if path == "/Items" and params.get("IncludeItemTypes") == "Playlist":
            items = [
                {"Id": pid, "Name": p["name"], "MediaType": p.get("media_type", "Audio"), "ChildCount": len(p["items"])}
                for pid, p in self.playlists.items()
            ]
            return httpx.Response(200, json={"Items": items, "TotalRecordCount": len(items)})
        if path == "/Items" and "Ids" in params:
            items = [t for t in self.tracks if t["Id"] == params["Ids"]]
            return httpx.Response(200, json={"Items": items, "TotalRecordCount": len(items)})
        if path == "/Items":
            start = int(params.get("StartIndex", 0))
            limit = int(params.get("Limit", 1000))
            page = self.tracks[start:start + limit]
            return httpx.Response(200, json={"Items": page, "TotalRecordCount": len(self.tracks)})
        if path == "/Playlists" and request.method == "POST":
            pid = f"{len(self.playlists) + 100:032x}"
            data = json.loads(request.content)
            self.playlists[pid] = {"name": data["Name"], "items": list(data["Ids"])}
            return httpx.Response(200, json={"Id": pid})
        if path.startswith("/Playlists/") and path.endswith("/Items"):
            pid = path.split("/")[2]
            playlist = self.playlists[pid]
            if request.method == "GET":
                items = [
                    {"Id": item_id, "PlaylistItemId": f"entry-{i}-{item_id}"}
                    for i, item_id in enumerate(playlist["items"])
                ]
                return httpx.Response(200, json={"Items": items, "TotalRecordCount": len(items)})
            if request.method == "POST":
                if self.fail_next_add:
                    self.fail_next_add = False
                    return httpx.Response(500)
                for item_id in params["Ids"].split(","):
                    if item_id not in playlist["items"]:  # Jellyfin skips duplicates
                        playlist["items"].append(item_id)
                return httpx.Response(204)
            if request.method == "DELETE":
                entry_ids = params["EntryIds"].split(",")
                playlist["items"] = [
                    item_id for i, item_id in enumerate(playlist["items"])
                    if f"entry-{i}-{item_id}" not in entry_ids
                ]
                return httpx.Response(204)
        return httpx.Response(404)


@pytest.fixture
def fake(monkeypatch):
    server = FakeJellyfin([_audio(n) for n in range(1, 2501)])
    real_client = httpx.Client

    def client_with_fake_transport(*args, **kwargs):
        return real_client(*args, transport=httpx.MockTransport(server.handler), **kwargs)

    monkeypatch.setattr(jellyfin_client.httpx, "Client", client_with_fake_transport)
    return server


@pytest.fixture
def client(fake):
    jf = JellyfinClient("http://jellyfin:8096", "token", "Music")
    assert jf.is_connected()
    return jf


class TestConnect:
    def test_finds_admin_user_and_music_library(self, client):
        assert client._user_id == USER_ID
        assert client._library_id == LIBRARY_ID


class TestTrackFetch:
    def test_sync_rows_page_through_library(self, client):
        rows = client.get_all_tracks_for_sync()

        assert len(rows) == 2500
        assert rows[0]["track"].title == "Track 1"
        assert rows[0]["album_key"] == ALBUM_ID
        assert rows[4]["view_count"] == 5
        assert rows[0]["last_viewed_at"] == "2026-01-02T03:04:05Z"

    def test_fetch_errors_raise(self, client, fake):
        """A failed fetch must raise so a sync keeps the previous cache."""
        fake.fail_paths.add("/Items")
        with pytest.raises(httpx.HTTPStatusError):
            client.get_all_tracks()

    def test_library_stats_counted_from_tracks(self, client, fake):
        fake.tracks = [
            _audio(1),
            _audio(2, Genres=["Jazz", "Blues"], ProductionYear=2003),
            _audio(3, ProductionYear=None),
        ]

        stats = client.get_library_stats()

        assert stats["total_tracks"] == 3
        assert stats["genres"] == [{"name": "Blues", "count": 1}, {"name": "Jazz", "count": 3}]
        assert stats["decades"] == [{"name": "1990s", "count": 1}, {"name": "2000s", "count": 1}]


class TestGetTrack:
    def test_looks_up_track_by_id(self, client):
        track = client.get_track_by_key(f"{7:032x}")
        assert track.title == "Track 7"

    def test_rejects_malformed_ids(self, client):
        assert client.get_track_by_key("../Users") is None


class TestArt:
    def test_uses_track_image_when_present(self, client, fake):
        fake.tracks = [_audio(1, ImageTags={"Primary": "x"})]
        assert client.get_art_url(f"{1:032x}") == f"http://jellyfin:8096/Items/{1:032x}/Images/Primary"

    def test_falls_back_to_album_image(self, client):
        assert client.get_art_url(f"{1:032x}") == f"http://jellyfin:8096/Items/{ALBUM_ID}/Images/Primary"

    def test_none_without_any_image(self, client, fake):
        fake.tracks = [_audio(1, AlbumPrimaryImageTag=None)]
        assert client.get_art_url(f"{1:032x}") is None

    def test_rejects_malformed_ids(self, client):
        assert client.get_art_url("../System/Info") is None


class TestUpdatePlaylist:
    def _playlist(self, client, items):
        return client.create_playlist("Mix", items)["playlist_id"]

    def test_append_skips_duplicates(self, client, fake):
        pid = self._playlist(client, ["a1", "a2"])

        result = client.update_playlist(pid, ["a2", "a3"], mode="append")

        assert result["tracks_added"] == 1
        assert result["duplicates_skipped"] == 1
        assert fake.playlists[pid]["items"] == ["a1", "a2", "a3"]

    def test_replace_keeps_overlapping_tracks(self, client, fake):
        pid = self._playlist(client, ["a1", "a2"])

        result = client.update_playlist(pid, ["a2", "a3"], mode="replace")

        assert result["success"] is True
        assert fake.playlists[pid]["items"] == ["a2", "a3"]

    def test_replace_restores_old_tracks_when_add_fails(self, client, fake):
        pid = self._playlist(client, ["a1", "a2"])
        fake.fail_next_add = True

        result = client.update_playlist(pid, ["a3"], mode="replace")

        assert result["success"] is False
        assert fake.playlists[pid]["items"] == ["a1", "a2"]

    def test_scratch_playlist_created_then_reused(self, client, fake):
        first = client.update_playlist("__scratch__", ["a1"], mode="replace")
        second = client.update_playlist("__scratch__", ["a2"], mode="append")

        assert first["success"] and second["success"]
        assert [p["name"] for p in fake.playlists.values()] == [SCRATCH_PLAYLIST_TITLE]
        assert next(iter(fake.playlists.values()))["items"] == ["a1", "a2"]

    def test_rejects_malformed_playlist_id(self, client):
        result = client.update_playlist("../Users", ["a1"], mode="append")
        assert result == {"success": False, "error": "Invalid playlist ID"}


class TestGetPlaylists:
    def test_lists_audio_and_imported_playlists_not_video(self, client, fake):
        fake.playlists = {
            "p1": {"name": "Mix", "items": ["a1", "a2"]},
            "p2": {"name": "Imported m3u", "items": ["a1"], "media_type": "Unknown"},
            "p3": {"name": "Movie night", "items": ["v1"], "media_type": "Video"},
        }

        playlists = client.get_playlists()

        assert [(p.title, p.track_count) for p in playlists] == [("Imported m3u", 1), ("Mix", 2)]


class TestSyncIntoCache:
    @pytest.fixture
    def temp_db(self, tmp_path, monkeypatch):
        monkeypatch.setattr(library_cache, "DB_PATH", tmp_path / "cache.db")
        monkeypatch.setattr(library_cache, "DATA_DIR", tmp_path)
        monkeypatch.setattr(library_cache, "_schema_initialized", False)
        monkeypatch.setattr(library_cache, "_sync_state", {
            "is_syncing": False, "phase": None, "current": 0, "total": 0, "error": None,
        })
        monkeypatch.setattr(JellyfinClient, "get_machine_identifier", lambda self: "jf-server")

    def test_sync_stores_album_and_play_history(self, client, temp_db):
        result = library_cache.sync_library(client)

        assert result["success"] is True
        albums = library_cache.get_album_candidates()
        assert len(albums) == 1
        assert albums[0]["parent_rating_key"] == ALBUM_ID
        assert albums[0]["track_count"] == 2500

    def test_failed_fetch_keeps_previous_cache(self, client, fake, temp_db):
        library_cache.sync_library(client)
        library_cache._sync_state["is_syncing"] = False
        fake.fail_paths.add("/Items")

        result = library_cache.sync_library(client)

        assert result["success"] is False
        assert library_cache.get_sync_state()["track_count"] == 2500
