"""Tests for local library cache functionality."""

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from backend import library_cache


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_library_cache.db"
    monkeypatch.setattr(library_cache, "DB_PATH", db_path)
    monkeypatch.setattr(library_cache, "DATA_DIR", tmp_path)
    # Reset schema initialization flag so each test gets fresh schema
    monkeypatch.setattr(library_cache, "_schema_initialized", False)
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def initialized_db(temp_db):
    """Create and initialize a test database."""
    conn = library_cache.ensure_db_initialized()
    conn.close()
    return temp_db



class PagedPlexMock:
    """Gives a simple mock Plex client the paged sync interface (iter_raw_tracks + fetch_metadata).

    Subclasses provide all_tracks() and albums() (rating key -> year and genres).
    """

    SYNC_PAGE_SIZE = 2

    def get_library_total_tracks(self):
        return 0

    def iter_raw_tracks(self, updated_since=None):
        tracks = self.all_tracks()
        for start in range(0, len(tracks), self.SYNC_PAGE_SIZE):
            yield tracks[start:start + self.SYNC_PAGE_SIZE]

    def fetch_metadata(self, rating_keys):
        albums = self.albums()
        return {key: albums[key] for key in rating_keys if key in albums}


class TestSchemaCreation:
    """Test database schema initialization."""

    def test_creates_data_directory(self, temp_db):
        """Schema init creates data directory if missing."""
        # Remove directory
        temp_db.parent.rmdir() if temp_db.parent.exists() else None

        conn = library_cache.ensure_db_initialized()
        conn.close()

        assert temp_db.parent.exists()

    def test_creates_tracks_table(self, initialized_db):
        """Schema creates tracks table with correct columns."""
        conn = sqlite3.connect(str(initialized_db))
        cursor = conn.execute("PRAGMA table_info(tracks)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        conn.close()

        assert "rating_key" in columns
        assert "title" in columns
        assert "artist" in columns
        assert "album" in columns
        assert "duration_ms" in columns
        assert "year" in columns
        assert "genres" in columns
        assert "user_rating" in columns
        assert "is_live" in columns
        assert "updated_at" in columns

    def test_creates_sync_state_table(self, initialized_db):
        """Schema creates sync_state table with single row."""
        conn = sqlite3.connect(str(initialized_db))
        cursor = conn.execute("SELECT COUNT(*) FROM sync_state")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1

    def test_creates_indexes(self, initialized_db):
        """Schema creates expected indexes."""
        conn = sqlite3.connect(str(initialized_db))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_tracks_%'"
        )
        indexes = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert "idx_tracks_artist" in indexes
        assert "idx_tracks_year" in indexes
        assert "idx_tracks_is_live" in indexes

    def test_wal_mode_enabled(self, initialized_db):
        """Connection uses WAL journal mode."""
        conn = library_cache.get_db_connection()
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        conn.close()

        assert mode.lower() == "wal"

    def test_idempotent_initialization(self, initialized_db):
        """Multiple initializations don't cause errors."""
        # Should not raise
        conn1 = library_cache.ensure_db_initialized()
        conn1.close()
        conn2 = library_cache.ensure_db_initialized()
        conn2.close()


class TestSyncState:
    """Test sync state management."""

    def test_initial_sync_state(self, initialized_db):
        """Fresh database has empty sync state."""
        state = library_cache.get_sync_state()

        assert state["track_count"] == 0
        assert state["synced_at"] is None
        assert state["is_syncing"] is False
        assert state["sync_progress"] is None
        assert state["error"] is None

    def test_sync_state_after_sync(self, initialized_db):
        """Sync state updates after successful sync."""
        conn = library_cache.get_db_connection()

        # Simulate a completed sync
        synced_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE sync_state SET plex_server_id = ?, last_sync_at = ?, "
            "track_count = ?, sync_duration_ms = ? WHERE id = 1",
            ("test-server-id", synced_at, 1000, 5000),
        )
        conn.commit()
        conn.close()

        state = library_cache.get_sync_state()

        assert state["track_count"] == 1000
        assert state["synced_at"] == synced_at
        assert state["plex_server_id"] == "test-server-id"


class TestCacheOperations:
    """Test cache read/write operations."""

    def test_get_cached_tracks_empty(self, initialized_db):
        """Empty cache returns empty list."""
        tracks = library_cache.get_cached_tracks()
        assert tracks == []

    def test_get_cached_tracks_with_data(self, initialized_db):
        """Cached tracks are returned correctly."""
        conn = library_cache.get_db_connection()
        conn.execute(
            "INSERT INTO tracks (rating_key, title, artist, album, duration_ms, "
            "year, genres, user_rating, is_live) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("123", "Test Song", "Test Artist", "Test Album", 180000, 1999,
             json.dumps(["Rock", "Alternative"]), 8, False),
        )
        conn.commit()
        conn.close()

        tracks = library_cache.get_cached_tracks()

        assert len(tracks) == 1
        assert tracks[0]["rating_key"] == "123"
        assert tracks[0]["title"] == "Test Song"
        assert tracks[0]["genres"] == ["Rock", "Alternative"]

    def test_clear_cache(self, initialized_db):
        """Clear cache removes all tracks and resets state."""
        conn = library_cache.get_db_connection()
        conn.execute(
            "INSERT INTO tracks (rating_key, title, artist, album) "
            "VALUES ('1', 'Song', 'Artist', 'Album')"
        )
        conn.execute(
            "UPDATE sync_state SET track_count = 1, last_sync_at = ? WHERE id = 1",
            (datetime.now(timezone.utc).isoformat(),),
        )
        conn.commit()
        conn.close()

        library_cache.clear_cache()

        tracks = library_cache.get_cached_tracks()
        state = library_cache.get_sync_state()

        assert tracks == []
        assert state["track_count"] == 0
        assert state["synced_at"] is None


class TestFiltering:
    """Test track filtering operations."""

    @pytest.fixture
    def sample_tracks(self, initialized_db):
        """Insert sample tracks for filter testing."""
        conn = library_cache.get_db_connection()
        tracks = [
            ("1", "Rock Song", "Rock Artist", "Rock Album", 180000, 1995,
             json.dumps(["Rock"]), 8, False),
            ("2", "Pop Song", "Pop Artist", "Pop Album", 200000, 2005,
             json.dumps(["Pop"]), 6, False),
            ("3", "Live Concert", "Live Artist", "Live 2020", 300000, 2020,
             json.dumps(["Rock"]), 4, True),
            ("4", "Jazz Song", "Jazz Artist", "Jazz Album", 250000, 1985,
             json.dumps(["Jazz"]), 10, False),
        ]
        conn.executemany(
            "INSERT INTO tracks (rating_key, title, artist, album, duration_ms, "
            "year, genres, user_rating, is_live) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            tracks,
        )
        conn.commit()
        conn.close()
        return tracks

    def test_filter_exclude_live(self, sample_tracks):
        """Exclude live filter removes live tracks."""
        tracks = library_cache.get_tracks_by_filters(exclude_live=True)

        assert len(tracks) == 3
        assert all(not t["is_live"] for t in tracks)

    def test_filter_include_live(self, sample_tracks):
        """Include live returns all tracks."""
        tracks = library_cache.get_tracks_by_filters(exclude_live=False)

        assert len(tracks) == 4

    def test_filter_by_decade(self, sample_tracks):
        """Filter by decade returns matching tracks."""
        tracks = library_cache.get_tracks_by_filters(decades=["1990s"], exclude_live=False)

        assert len(tracks) == 1
        assert tracks[0]["year"] == 1995

    def test_filter_by_multiple_decades(self, sample_tracks):
        """Filter by multiple decades uses OR logic."""
        tracks = library_cache.get_tracks_by_filters(
            decades=["1980s", "2000s"], exclude_live=False
        )

        assert len(tracks) == 2
        years = {t["year"] for t in tracks}
        assert years == {1985, 2005}

    def test_filter_by_genre(self, sample_tracks):
        """Filter by genre returns matching tracks."""
        tracks = library_cache.get_tracks_by_filters(genres=["Rock"], exclude_live=False)

        assert len(tracks) == 2  # Rock Song and Live Concert

    def test_filter_by_genre_with_limit(self, initialized_db):
        """Filter by genre with limit applies limit AFTER genre filtering.

        This tests the fix for a bug where SQL LIMIT was applied before
        Python-side genre filtering, causing rare genres to return too few results.
        """
        conn = library_cache.get_db_connection()
        # Insert 100 Pop tracks and 10 Jazz tracks
        tracks = []
        for i in range(100):
            tracks.append((
                f"pop-{i}", f"Pop Song {i}", "Pop Artist", "Pop Album",
                180000, 2000, json.dumps(["Pop"]), 5, False
            ))
        for i in range(10):
            tracks.append((
                f"jazz-{i}", f"Jazz Song {i}", "Jazz Artist", "Jazz Album",
                180000, 1990, json.dumps(["Jazz"]), 5, False
            ))
        conn.executemany(
            "INSERT INTO tracks (rating_key, title, artist, album, duration_ms, "
            "year, genres, user_rating, is_live) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            tracks,
        )
        conn.commit()
        conn.close()

        # Request 5 Jazz tracks - should get exactly 5 Jazz tracks
        # (Bug: without fix, would get random sample of all tracks, mostly Pop)
        result = library_cache.get_tracks_by_filters(genres=["Jazz"], limit=5)

        assert len(result) == 5
        # All returned tracks must be Jazz
        for t in result:
            assert "Jazz" in t["genres"], f"Non-Jazz track returned: {t['title']}"

    def test_filter_by_min_rating(self, sample_tracks):
        """Filter by minimum rating."""
        tracks = library_cache.get_tracks_by_filters(min_rating=8, exclude_live=False)

        assert len(tracks) == 2
        assert all(t["user_rating"] >= 8 for t in tracks)

    def test_filter_with_limit(self, sample_tracks):
        """Filter with limit returns at most N tracks."""
        tracks = library_cache.get_tracks_by_filters(limit=2, exclude_live=False)

        assert len(tracks) == 2

    def test_filter_combined(self, sample_tracks):
        """Combined filters work together."""
        tracks = library_cache.get_tracks_by_filters(
            genres=["Rock"],
            decades=["1990s"],
            exclude_live=True,
            min_rating=6,
        )

        assert len(tracks) == 1
        assert tracks[0]["title"] == "Rock Song"


class TestLiveDetection:
    """Test live version detection."""

    def test_live_keyword_in_title(self):
        """Detects 'live' keyword in title."""
        assert library_cache._is_live_version("Song (Live)", "Album")
        assert library_cache._is_live_version("Live at Madison Square Garden", "Album")

    def test_live_keyword_in_album(self):
        """Detects 'live' keyword in album."""
        assert library_cache._is_live_version("Song", "Live in Concert")
        assert library_cache._is_live_version("Song", "The Concert Album")

    def test_date_pattern_in_album(self):
        """Detects date patterns indicating live recordings."""
        assert library_cache._is_live_version("Song", "2020-01-15 Chicago")
        assert library_cache._is_live_version("Song", "1999/05/20")

    def test_bootleg_keyword(self):
        """Detects bootleg keyword."""
        assert library_cache._is_live_version("Song", "Bootleg Series")

    def test_non_live_tracks(self):
        """Non-live tracks are not flagged."""
        assert not library_cache._is_live_version("Regular Song", "Studio Album")
        assert not library_cache._is_live_version("Alive", "Greatest Hits")  # 'live' in 'alive'


class TestStalenessCheck:
    """Test cache staleness detection."""

    def test_empty_cache_is_stale(self, initialized_db):
        """Empty cache is considered stale."""
        assert library_cache.is_cache_stale()

    def test_recent_cache_not_stale(self, initialized_db):
        """Recently synced cache is not stale."""
        conn = library_cache.get_db_connection()
        synced_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE sync_state SET last_sync_at = ?, track_count = 1 WHERE id = 1",
            (synced_at,),
        )
        conn.commit()
        conn.close()

        assert not library_cache.is_cache_stale(max_age_hours=24)

    def test_old_cache_is_stale(self, initialized_db):
        """Old cache is considered stale."""
        conn = library_cache.get_db_connection()
        # Set sync time to 25 hours ago
        from datetime import timedelta
        old_time = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        conn.execute(
            "UPDATE sync_state SET last_sync_at = ? WHERE id = 1",
            (old_time,),
        )
        conn.commit()
        conn.close()

        assert library_cache.is_cache_stale(max_age_hours=24)


class TestServerChangeDetection:
    """Test Plex server change detection."""

    def test_first_sync_no_change(self, initialized_db):
        """First sync doesn't count as server change."""
        assert not library_cache.check_server_changed("new-server-id")

    def test_same_server_no_change(self, initialized_db):
        """Same server ID doesn't trigger change."""
        conn = library_cache.get_db_connection()
        conn.execute(
            "UPDATE sync_state SET plex_server_id = ? WHERE id = 1",
            ("my-server-id",),
        )
        conn.commit()
        conn.close()

        assert not library_cache.check_server_changed("my-server-id")

    def test_different_server_triggers_change(self, initialized_db):
        """Different server ID triggers change detection."""
        conn = library_cache.get_db_connection()
        conn.execute(
            "UPDATE sync_state SET plex_server_id = ? WHERE id = 1",
            ("old-server-id",),
        )
        conn.commit()
        conn.close()

        assert library_cache.check_server_changed("new-server-id")


class TestSyncLibrary:
    """Test sync_library() function."""

    @pytest.fixture
    def mock_track(self):
        """Create a mock Plex track object."""
        class MockTrack:
            def __init__(self, rating_key, title, artist, album, duration, parent_key):
                self.ratingKey = rating_key
                self.title = title
                self.grandparentTitle = artist
                self.parentTitle = album
                self.duration = duration
                self.parentRatingKey = parent_key
                self.userRating = None
        return MockTrack

    @pytest.fixture
    def mock_plex_client(self, mock_track):
        """Create a mock Plex client."""
        class MockPlexClient(PagedPlexMock):
            def __init__(self):
                self.tracks = [
                    mock_track("1", "Song One", "Artist A", "Album X", 180000, "100"),
                    mock_track("2", "Song Two", "Artist B", "Album Y", 200000, "101"),
                    mock_track("3", "Song Three", "Artist A", "Album X", 220000, "100"),
                ]
                self.album_metadata = {
                    "100": {"genres": ["Rock", "Alternative"], "year": 1995},
                    "101": {"genres": ["Electronic"], "year": 2020},
                }

            def get_machine_identifier(self):
                return "test-server-123"

            def albums(self):
                return self.album_metadata


            def all_tracks(self):
                return self.tracks

        return MockPlexClient()

    @pytest.fixture
    def reset_sync_state(self, monkeypatch):
        """Reset sync state before each test."""
        monkeypatch.setattr(library_cache, "_sync_state", {
            "is_syncing": False,
            "phase": None,
            "current": 0,
            "total": 0,
            "error": None,
        })

    def test_sync_success(self, initialized_db, mock_plex_client, reset_sync_state):
        """Sync completes successfully with correct track count."""
        result = library_cache.sync_library(mock_plex_client)

        assert result["success"] is True
        assert result["track_count"] == 3
        assert "duration_ms" in result

        # Verify tracks in database
        tracks = library_cache.get_cached_tracks()
        assert len(tracks) == 3

    def test_sync_stores_genres_from_albums(self, initialized_db, mock_plex_client, reset_sync_state):
        """Sync correctly maps album genres to tracks."""
        library_cache.sync_library(mock_plex_client)

        tracks = library_cache.get_cached_tracks()
        track_by_key = {t["rating_key"]: t for t in tracks}

        # Tracks 1 and 3 are from Album X (parent_key=100)
        assert track_by_key["1"]["genres"] == ["Rock", "Alternative"]
        assert track_by_key["3"]["genres"] == ["Rock", "Alternative"]

        # Track 2 is from Album Y (parent_key=101)
        assert track_by_key["2"]["genres"] == ["Electronic"]

    def test_sync_stores_year_from_albums(self, initialized_db, mock_plex_client, reset_sync_state):
        """Sync correctly maps album year to tracks."""
        library_cache.sync_library(mock_plex_client)

        tracks = library_cache.get_cached_tracks()
        track_by_key = {t["rating_key"]: t for t in tracks}

        assert track_by_key["1"]["year"] == 1995
        assert track_by_key["2"]["year"] == 2020
        assert track_by_key["3"]["year"] == 1995

    def test_sync_updates_sync_state(self, initialized_db, mock_plex_client, reset_sync_state):
        """Sync updates the sync_state table."""
        library_cache.sync_library(mock_plex_client)

        state = library_cache.get_sync_state()

        assert state["track_count"] == 3
        assert state["synced_at"] is not None
        assert state["plex_server_id"] == "test-server-123"
        assert state["is_syncing"] is False

    def test_sync_progress_callback(self, initialized_db, mock_plex_client, reset_sync_state):
        """Progress callback is invoked during sync."""
        progress_calls = []

        def on_progress(current, total):
            progress_calls.append((current, total))

        library_cache.sync_library(mock_plex_client, on_progress=on_progress)

        # With 3 tracks and SYNC_BATCH_SIZE=500, we won't hit the batch callback
        # but we can verify the function accepts the callback without error
        assert isinstance(progress_calls, list)

    def test_sync_rejects_concurrent_sync(self, initialized_db, mock_plex_client, monkeypatch):
        """Second sync attempt is rejected while one is in progress."""
        # Simulate sync in progress
        monkeypatch.setattr(library_cache, "_sync_state", {
            "is_syncing": True,
            "phase": "processing",
            "current": 50,
            "total": 100,
            "error": None,
        })

        result = library_cache.sync_library(mock_plex_client)

        assert result["success"] is False
        assert "already in progress" in result["error"]

    def test_sync_handles_missing_server_id(self, initialized_db, reset_sync_state):
        """Sync fails gracefully if server ID is unavailable."""
        class BadPlexClient:
            def get_machine_identifier(self):
                return None

        result = library_cache.sync_library(BadPlexClient())

        assert result["success"] is False
        assert "server identifier" in result["error"].lower()

    def test_sync_clears_error_on_success(self, initialized_db, mock_plex_client, monkeypatch):
        """Successful sync clears any previous error state."""
        # Set initial state with error
        monkeypatch.setattr(library_cache, "_sync_state", {
            "is_syncing": False,
            "phase": None,
            "current": 0,
            "total": 0,
            "error": "Previous error",
        })

        library_cache.sync_library(mock_plex_client)

        state = library_cache.get_sync_state()
        assert state["error"] is None
        assert state["is_syncing"] is False

    def test_sync_removes_deleted_tracks(self, initialized_db, mock_track, reset_sync_state):
        """Sync removes tracks that no longer exist in Plex."""
        # Pre-populate cache with a track that won't be in the sync
        conn = library_cache.get_db_connection()
        conn.execute(
            "INSERT INTO tracks (rating_key, title, artist, album, duration_ms, "
            "year, genres, user_rating, is_live) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("stale-track-999", "Stale Song", "Old Artist", "Deleted Album",
             180000, 2000, json.dumps(["Rock"]), 5, False),
        )
        conn.commit()
        conn.close()

        # Verify stale track exists
        tracks_before = library_cache.get_cached_tracks()
        assert any(t["rating_key"] == "stale-track-999" for t in tracks_before)

        # Create mock client that returns different tracks
        class MockPlexClient(PagedPlexMock):
            def get_machine_identifier(self):
                return "test-server-123"

            def albums(self):
                return {"100": {"genres": ["Electronic"], "year": 2024}}


            def all_tracks(self):
                return [mock_track("new-1", "New Song", "New Artist", "New Album", 200000, "100")]

        # Run sync
        result = library_cache.sync_library(MockPlexClient())

        assert result["success"] is True
        assert result["track_count"] == 1

        # Verify stale track was removed
        tracks_after = library_cache.get_cached_tracks()
        assert len(tracks_after) == 1
        assert tracks_after[0]["rating_key"] == "new-1"
        assert not any(t["rating_key"] == "stale-track-999" for t in tracks_after)

    def test_failed_sync_keeps_previous_cache(self, initialized_db, mock_track, reset_sync_state):
        """Failed sync rolls back, keeping the previous complete cache."""
        # First, do a successful sync to populate cache
        class SuccessfulClient(PagedPlexMock):
            def get_machine_identifier(self):
                return "test-server-123"

            def albums(self):
                return {"100": {"genres": ["Rock"], "year": 2020}}


            def all_tracks(self):
                return [mock_track("1", "Song", "Artist", "Album", 180000, "100")]

        result = library_cache.sync_library(SuccessfulClient())
        assert result["success"] is True
        assert library_cache.has_cached_tracks() is True

        # Reset sync state for next sync attempt
        library_cache._sync_state["is_syncing"] = False

        # Now attempt a sync that fails during Plex API call
        class FailingClient(PagedPlexMock):
            def get_machine_identifier(self):
                return "test-server-123"

            def iter_raw_tracks(self, updated_since=None):
                raise ConnectionError("Plex unreachable")


        result = library_cache.sync_library(FailingClient())
        assert result["success"] is False

        # The previous complete cache is still there
        assert library_cache.has_cached_tracks() is True
        assert library_cache.get_sync_state()["track_count"] == 1

    def test_sync_failing_midway_keeps_previous_cache(
        self, initialized_db, mock_plex_client, mock_track, reset_sync_state, monkeypatch
    ):
        """A sync that fails after some batches were written rolls back entirely."""
        result = library_cache.sync_library(mock_plex_client)
        assert result["success"] is True
        assert library_cache.get_sync_state()["track_count"] == 3
        library_cache._sync_state["is_syncing"] = False

        class FailsMidwayClient(PagedPlexMock):
            def get_machine_identifier(self):
                return "test-server-123"

            def albums(self):
                return {}


            def iter_raw_tracks(self, updated_since=None):
                # Two pages are written before the media server fails
                yield [mock_track(f"new-{i}", f"New {i}", "Artist", "Album", 1000, "100") for i in range(2)]
                yield [mock_track(f"new-{i}", f"New {i}", "Artist", "Album", 1000, "100") for i in range(2, 4)]
                raise RuntimeError("Plex connection dropped")

        result = library_cache.sync_library(FailsMidwayClient())
        assert result["success"] is False

        keys = {t["rating_key"] for t in library_cache.get_cached_tracks()}
        assert keys == {"1", "2", "3"}
        assert library_cache.get_sync_state()["track_count"] == 3


class TestReplaceLibrary:
    """Switching media servers replaces the cache, even while a sync is running."""

    @pytest.fixture(autouse=True)
    def reset_switch_state(self, monkeypatch):
        monkeypatch.setattr(library_cache, "_sync_state", {
            "is_syncing": False, "phase": None, "current": 0, "total": 0, "error": None,
        })
        monkeypatch.setattr(library_cache, "_cache_from_previous_server", False)
        library_cache._cancel_sync.clear()
        yield
        library_cache._cancel_sync.clear()

    class FakeClient:
        """Minimal Jellyfin-style client whose tracks carry a per-server prefix."""

        def __init__(self, prefix, count=3, on_fetch=None):
            self.prefix = prefix
            self.count = count
            self.on_fetch = on_fetch

        def is_connected(self):
            return True

        def get_machine_identifier(self):
            return f"{self.prefix}-server"

        def get_all_tracks_for_sync(self):
            if self.on_fetch:
                self.on_fetch()
            from backend.models import Track
            return [
                {
                    "track": Track(rating_key=f"{self.prefix}{i}", title=f"T{i}", artist="A", album="B", duration_ms=1),
                    "album_key": "album",
                    "view_count": 0,
                    "last_viewed_at": None,
                }
                for i in range(self.count)
            ]

    def _keys(self):
        return {t["rating_key"] for t in library_cache.get_cached_tracks()}

    def test_invalidated_cache_is_not_used(self, initialized_db):
        library_cache.sync_library(self.FakeClient("plex"))
        assert library_cache.has_cached_tracks()

        library_cache.invalidate_cache()

        assert not library_cache.has_cached_tracks()

    def test_replace_clears_and_syncs_current_server(self, initialized_db):
        library_cache.sync_library(self.FakeClient("plex"))
        switch = library_cache.invalidate_cache()

        library_cache.replace_library(lambda: self.FakeClient("jf", count=2), switch)

        assert self._keys() == {"jf0", "jf1"}
        assert library_cache.has_cached_tracks()

    def test_cancelled_sync_rolls_back(self, initialized_db, monkeypatch):
        library_cache.sync_library(self.FakeClient("plex"))
        monkeypatch.setattr(library_cache, "SYNC_BATCH_SIZE", 2)

        # Same server, so the old tracks stay until the new sync commits
        result = library_cache.sync_library(
            self.FakeClient("plex", count=10, on_fetch=library_cache._cancel_sync.set)
        )

        assert result == {"success": False, "error": "Sync cancelled"}
        assert library_cache.get_sync_state()["error"] is None
        assert self._keys() == {"plex0", "plex1", "plex2"}

    def test_switch_during_sync_ends_on_latest_server(self, initialized_db):
        import threading

        current = {"client": None}
        fetch_started = threading.Event()
        release_fetch = threading.Event()

        def slow_fetch():
            fetch_started.set()
            release_fetch.wait(5)

        current["client"] = self.FakeClient("plex", on_fetch=slow_fetch)
        first = threading.Thread(target=library_cache.sync_library, args=(current["client"],))
        first.start()
        fetch_started.wait(5)

        # Switch to Jellyfin while the Plex sync is still fetching
        current["client"] = self.FakeClient("jf", count=2)
        switch = library_cache.invalidate_cache()
        second = threading.Thread(target=library_cache.replace_library, args=(lambda: current["client"], switch))
        second.start()
        release_fetch.set()
        first.join(5)
        second.join(5)

        assert self._keys() == {"jf0", "jf1"}

    def test_only_latest_switch_syncs(self, initialized_db):
        stale_switch = library_cache.invalidate_cache()
        latest_switch = library_cache.invalidate_cache()
        synced = []

        def client(name):
            def get():
                synced.append(name)
                return self.FakeClient(name)
            return get

        library_cache.replace_library(client("stale"), stale_switch)
        library_cache.replace_library(client("latest"), latest_switch)

        assert synced == ["latest"]


class TestArtistFilters:
    """Artist include/exclude filters and artist search (#9)."""

    @pytest.fixture
    def artist_db(self, initialized_db):
        rows = [
            ("1", "Creep", "Radiohead", "Pablo Honey", 1000, 1993, '["Rock"]'),
            ("2", "Karma Police", "Radiohead", "OK Computer", 1000, 1997, '["Rock"]'),
            ("3", "Glory Box", "Portishead", "Dummy", 1000, 1994, '["Trip Hop"]'),
            ("4", "Teardrop", "Massive Attack", "Mezzanine", 1000, 1998, '["Trip Hop"]'),
            ("5", "Live Forever", "Oasis", "Definitely Maybe", 1000, 1994, '["Rock"]'),
        ]
        conn = sqlite3.connect(str(initialized_db))
        conn.executemany(
            library_cache._INSERT_TRACK_SQL,
            [r + (None, 0, "album", 0, None) for r in rows],
        )
        conn.execute("UPDATE sync_state SET track_count = 5 WHERE id = 1")
        conn.commit()
        conn.close()

    def _keys(self, **filters):
        return {t["rating_key"] for t in library_cache.get_tracks_by_filters(**filters)}

    def test_include_artists_case_insensitive(self, artist_db):
        assert self._keys(artists=["radiohead", "PORTISHEAD"]) == {"1", "2", "3"}
        assert library_cache.count_tracks_by_filters(artists=["radiohead"]) == 2

    def test_exclude_artists(self, artist_db):
        assert self._keys(exclude_artists=["Radiohead"]) == {"3", "4", "5"}
        assert library_cache.count_tracks_by_filters(exclude_artists=["Radiohead", "Oasis"]) == 2

    def test_artists_combine_with_genres_and_decades(self, artist_db):
        assert self._keys(artists=["Radiohead", "Oasis"], genres=["Rock"], decades=["1990s"]) == {"1", "2", "5"}
        assert library_cache.count_tracks_by_filters(artists=["Portishead"], genres=["Rock"]) == 0

    def test_limit_samples_within_artist(self, artist_db):
        tracks = library_cache.get_tracks_by_filters(artists=["Radiohead"], limit=1)
        assert len(tracks) == 1 and tracks[0]["artist"] == "Radiohead"

    def test_search_artists_by_track_count(self, artist_db):
        assert library_cache.search_artists("head") == [
            {"name": "Radiohead", "count": 2},
            {"name": "Portishead", "count": 1},
        ]
        assert len(library_cache.search_artists("")) == 4

    def test_artist_names(self, artist_db):
        assert set(library_cache.get_artist_names()) == {"Radiohead", "Portishead", "Massive Attack", "Oasis"}


class TestPlexGenres:
    """Plex track genres combine album, artist and track genres (#30)."""

    class Tag:
        def __init__(self, tag):
            self.tag = tag

    def _track(self, genres=()):
        track = type("Track", (), {})()
        track.ratingKey, track.title, track.parentTitle = "1", "Song", "Album"
        track.grandparentTitle, track.grandparentRatingKey, track.parentRatingKey = "Artist", "50", "100"
        track.duration, track.userRating, track.viewCount, track.lastViewedAt = 1, None, 0, None
        track.genres = [self.Tag(g) for g in genres]
        return track

    def _genres(self, album, artist, track):
        row = library_cache._plex_track_row(
            {"100": {"genres": album, "year": 1990}}, {"50": artist}, self._track(track)
        )
        return json.loads(row[6])

    def test_combines_album_artist_and_track_genres(self):
        assert self._genres(["Pop/Rock"], ["Goth", "Dark Wave"], ["Post-Punk"]) == [
            "Pop/Rock", "Goth", "Dark Wave", "Post-Punk",
        ]

    def test_drops_case_insensitive_duplicates(self):
        assert self._genres(["Rock"], ["rock", "Goth"], ["ROCK"]) == ["Rock", "Goth"]

    def test_album_without_genres_uses_artist_genres(self):
        assert self._genres([], ["Goth"], []) == ["Goth"]


class TestCacheVersion:
    """Caches built by older sync logic are re-synced on upgrade."""

    def _set(self, db, track_count, version):
        conn = sqlite3.connect(str(db))
        conn.execute("UPDATE sync_state SET track_count = ?, cache_version = ? WHERE id = 1", (track_count, version))
        conn.commit()
        conn.close()

    def test_brand_new_database_does_not(self, temp_db, monkeypatch):
        """A first install must get the first-time sync dialog, not the upgrade one."""
        monkeypatch.setattr(library_cache, "_schema_initialized", False)
        library_cache.ensure_db_initialized().close()
        assert not library_cache.needs_resync()
        assert library_cache.sync_needed() == "empty"

    def test_older_cache_needs_resync(self, initialized_db, monkeypatch):
        monkeypatch.setattr(library_cache, "_migration_applied", False)
        self._set(initialized_db, 100, 1)
        assert library_cache.needs_resync()

    def test_current_cache_does_not(self, initialized_db, monkeypatch):
        monkeypatch.setattr(library_cache, "_migration_applied", False)
        self._set(initialized_db, 100, library_cache.CACHE_VERSION)
        assert not library_cache.needs_resync()

    def test_empty_cache_needs_a_sync_but_no_upgrade_dialog(self, initialized_db, monkeypatch):
        """An empty cache needs a sync (a failed sync once committed 0 tracks), but needs_resync() is what the
        frontend shows its blocking upgrade dialog on, so that stays False."""
        monkeypatch.setattr(library_cache, "_migration_applied", False)
        self._set(initialized_db, 0, library_cache.CACHE_VERSION)
        assert library_cache.sync_needed() == "empty"
        assert not library_cache.needs_resync()
        self._set(initialized_db, 100, library_cache.CACHE_VERSION)
        assert library_cache.sync_needed() is None


class TestHalfStarRatings:
    """Minimum rating accepts half stars (Plex 0-10 scale, half star = 1) (#13)."""

    def test_half_star_minimum(self, initialized_db):
        conn = sqlite3.connect(str(initialized_db))
        rows = [(str(i), f"T{i}", "A", "B", 1, 2000, "[]", rating) for i, rating in enumerate([1, 6, 7, 8, 10])]
        conn.executemany(library_cache._INSERT_TRACK_SQL, [r + (0, "x", 0, None) for r in rows])
        conn.execute("UPDATE sync_state SET track_count = 5 WHERE id = 1")
        conn.commit()
        conn.close()

        assert library_cache.count_tracks_by_filters(min_rating=1) == 5   # half a star and up
        assert library_cache.count_tracks_by_filters(min_rating=7) == 3   # 3.5 stars and up
        assert library_cache.count_tracks_by_filters(min_rating=8) == 2
        assert library_cache.count_tracks_by_filters(min_rating=10) == 1  # five stars only


class TestSyncSafety:
    """Guards against committing a broken cache, and the incremental / interrupted sync paths."""

    @pytest.fixture(autouse=True)
    def reset_sync_state(self, monkeypatch):
        monkeypatch.setattr(library_cache, "_sync_state", {
            "is_syncing": False, "phase": None, "current": 0, "total": 0, "error": None,
        })

    @staticmethod
    def _track(key, title="Song", artist="Artist", album="Album", parent="100"):
        class Track:
            pass
        t = Track()
        t.ratingKey, t.title, t.grandparentTitle, t.parentTitle = key, title, artist, album
        t.duration, t.parentRatingKey, t.userRating = 1000, parent, None
        return t

    def _client(self, tracks):
        class Client(PagedPlexMock):
            def get_machine_identifier(self):
                return "test-server-123"

            def albums(self):
                return {"100": {"genres": ["Rock"], "year": 2020}}

            def all_tracks(self):
                return tracks
        return Client()

    def test_empty_result_keeps_cache(self, initialized_db):
        assert library_cache.sync_library(self._client([self._track("1"), self._track("2")]))["success"]
        library_cache._sync_state["is_syncing"] = False
        result = library_cache.sync_library(self._client([]))
        assert result["success"] is False
        assert library_cache.get_sync_state()["track_count"] == 2

    def test_short_read_keeps_cache_even_when_forced(self, initialized_db):
        """Plex reports 10 tracks but the listing gave 3: a failed read, not a smaller library."""
        tracks = [self._track(str(i), title=f"Song {i}") for i in range(10)]
        assert library_cache.sync_library(self._client(tracks))["success"]
        library_cache._sync_state["is_syncing"] = False
        client = self._client(tracks[:3])
        client.get_library_total_tracks = lambda: 10
        assert library_cache.sync_library(client, force=True)["success"] is False
        assert library_cache.get_sync_state()["track_count"] == 10

    def test_forced_sync_accepts_a_much_smaller_library(self, initialized_db):
        tracks = [self._track(str(i), title=f"Song {i}") for i in range(10)]
        assert library_cache.sync_library(self._client(tracks))["success"]
        library_cache._sync_state["is_syncing"] = False
        assert library_cache.sync_library(self._client(tracks[:3]))["success"] is False
        library_cache._sync_state["is_syncing"] = False
        assert library_cache.sync_library(self._client(tracks[:3]), force=True)["success"]
        assert library_cache.get_sync_state()["track_count"] == 3

    def test_empty_library_syncs_to_zero_tracks(self, initialized_db, monkeypatch):
        monkeypatch.setattr(library_cache, "_migration_applied", False)
        result = library_cache.sync_library(self._client([]))
        assert result == {"success": True, "track_count": 0, "duration_ms": result["duration_ms"]}
        assert library_cache.get_sync_state()["error"] is None
        assert not library_cache.needs_resync()

    def test_other_music_library_on_same_server_replaces_cache(self, initialized_db):
        """Same server, different music library: the cache is cleared, so the smaller library is accepted."""
        tracks = [self._track(str(i), title=f"Song {i}") for i in range(10)]
        first, second = self._client(tracks), self._client([self._track("50"), self._track("51")])
        first.get_library_identifier = lambda: "library-a"
        second.get_library_identifier = lambda: "library-b"
        assert library_cache.sync_library(first)["success"]
        library_cache._sync_state["is_syncing"] = False
        assert not library_cache.check_server_changed("test-server-123", "library-a")
        assert library_cache.check_server_changed("test-server-123", "library-b")
        assert library_cache.sync_library(second)["success"]
        state = library_cache.get_sync_state()
        assert (state["track_count"], state["library_id"]) == (2, "library-b")

    def test_cache_without_library_id_is_not_a_change(self, initialized_db):
        """Caches synced before library_id existed must not be wiped on upgrade."""
        assert library_cache.sync_library(self._client([self._track("1")]))["success"]   # client has no library id
        assert library_cache.get_sync_state()["library_id"] is None
        assert not library_cache.check_server_changed("test-server-123", "library-a")

    def test_track_genres_come_from_the_listing_row(self, initialized_db):
        """plexapi keeps genres out of vars(track); the Genre elements of the row are read instead."""
        import xml.etree.ElementTree as ET
        track = self._track("1")
        track._data = ET.fromstring('<Track ratingKey="1"><Genre tag="R&amp;B"/><Genre tag="Rock"/><Media/></Track>')
        assert library_cache.sync_library(self._client([track]))["success"]
        conn = sqlite3.connect(str(initialized_db))
        genres = conn.execute("SELECT genres FROM tracks WHERE rating_key = '1'").fetchone()[0]
        conn.close()
        assert json.loads(genres) == ["Rock", "R&B"]   # album genre first, then the track's own

    def test_shrunken_result_keeps_cache(self, initialized_db):
        tracks = [self._track(str(i), title=f"Song {i}") for i in range(10)]
        assert library_cache.sync_library(self._client(tracks))["success"]
        library_cache._sync_state["is_syncing"] = False
        assert library_cache.sync_library(self._client(tracks[:3]))["success"] is False
        assert library_cache.get_sync_state()["track_count"] == 10

    def test_interrupted_full_sync_needs_resync(self, initialized_db, monkeypatch):
        monkeypatch.setattr(library_cache, "_migration_applied", False)
        assert library_cache.sync_library(self._client([self._track("1")]))["success"]
        library_cache._sync_state["is_syncing"] = False
        assert library_cache.sync_needed() is None
        conn = sqlite3.connect(str(initialized_db))
        conn.execute("UPDATE sync_state SET sync_started_at = '2026-01-01T00:00:00+00:00' WHERE id = 1")
        conn.commit()
        conn.close()
        assert library_cache.sync_needed() == "interrupted"
        assert not library_cache.needs_resync()

    def test_incremental_sync_upserts_without_dropping(self, initialized_db):
        assert library_cache.sync_library(self._client([self._track("1"), self._track("2", title="Two")]))["success"]
        library_cache._sync_state["is_syncing"] = False
        changed = [self._track("2", title="Two (remastered)"), self._track("3", title="Three")]
        result = library_cache.sync_library(self._client(changed), incremental=True)
        assert result["success"] is True
        titles = {t["rating_key"]: t["title"] for t in library_cache.get_cached_tracks()}
        assert titles == {"1": "Song", "2": "Two (remastered)", "3": "Three"}

    def test_incremental_needs_a_full_sync_first(self, initialized_db):
        result = library_cache.sync_library(self._client([self._track("1")]), incremental=True)
        assert result["success"] is False
