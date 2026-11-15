"""Tests for Plex client."""

import time
from unittest.mock import MagicMock, patch

from requests.exceptions import ConnectionError as RequestsConnectionError

from backend.models import PlexPlaylistInfo, Track
from backend.plex_client import TrackCache


class TestPlexClientConnection:
    """Tests for Plex connection handling."""

    def test_connect_with_valid_credentials(self, mocker):
        """Should connect successfully with valid credentials."""
        from backend.plex_client import PlexClient

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://localhost:32400", "valid-token", "Music")
            assert client.is_connected() is True

    def test_connect_with_invalid_token(self, mocker):
        """Should handle invalid token gracefully."""
        from backend.plex_client import PlexClient
        from plexapi.exceptions import Unauthorized

        with patch("backend.plex_client.PlexServer", side_effect=Unauthorized("Invalid token")):
            client = PlexClient("http://localhost:32400", "invalid-token", "Music")
            assert client.is_connected() is False
            assert "invalid" in client.get_error().lower() or "unauthorized" in client.get_error().lower()

    def test_connect_with_unreachable_server(self, mocker):
        """Should handle unreachable server gracefully."""
        from backend.plex_client import PlexClient
        from requests.exceptions import ConnectionError

        with patch("backend.plex_client.PlexServer", side_effect=ConnectionError("Connection refused")):
            client = PlexClient("http://unreachable:32400", "token", "Music")
            assert client.is_connected() is False
            assert client.get_error() is not None

    def test_connect_with_missing_library(self, mocker):
        """Should handle missing music library."""
        from backend.plex_client import PlexClient
        from plexapi.exceptions import NotFound

        mock_server = MagicMock()
        mock_server.library.section.side_effect = NotFound("Library not found")

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://localhost:32400", "token", "NonexistentLibrary")
            assert client.is_connected() is False
            assert "library" in client.get_error().lower() or "not found" in client.get_error().lower()


class TestPlexClientReconnect:
    """Tests for lazy reconnection with cooldown."""

    def test_reconnect_attempted_when_disconnected(self, mocker):
        """Should attempt reconnection when disconnected and cooldown passed."""
        from backend.plex_client import PlexClient
        from requests.exceptions import ConnectionError

        # First connection fails
        with patch("backend.plex_client.PlexServer", side_effect=ConnectionError("Connection refused")):
            client = PlexClient("http://localhost:32400", "token", "Music")
            assert client.is_connected() is False

        # Now server is available - mock successful connection
        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            # Force cooldown to be passed
            client._last_reconnect_attempt = 0
            assert client.is_connected() is True

    def test_reconnect_respects_cooldown(self, mocker):
        """Should not attempt reconnection if cooldown hasn't passed."""
        from backend.plex_client import PlexClient
        from requests.exceptions import ConnectionError

        connect_count = 0
        def mock_connect(*args, **kwargs):
            nonlocal connect_count
            connect_count += 1
            raise ConnectionError("Connection refused")

        with patch("backend.plex_client.PlexServer", side_effect=mock_connect):
            client = PlexClient("http://localhost:32400", "token", "Music")
            initial_count = connect_count

            # Multiple is_connected() calls within cooldown should not retry
            client.is_connected()
            client.is_connected()
            client.is_connected()

            # Should only have the initial connection attempt
            assert connect_count == initial_count

    def test_reconnect_after_cooldown(self, mocker):
        """Should retry connection after cooldown period."""
        from backend.plex_client import PlexClient
        from requests.exceptions import ConnectionError

        connect_count = 0
        def mock_connect(*args, **kwargs):
            nonlocal connect_count
            connect_count += 1
            raise ConnectionError("Connection refused")

        with patch("backend.plex_client.PlexServer", side_effect=mock_connect):
            client = PlexClient("http://localhost:32400", "token", "Music")
            initial_count = connect_count

            # Simulate cooldown passed
            client._last_reconnect_attempt = time.time() - client.RECONNECT_COOLDOWN - 1
            client.is_connected()

            # Should have attempted another connection
            assert connect_count == initial_count + 1


class TestPlexClientLibraryStats:
    """Tests for library statistics."""

    def test_get_library_stats(self, mocker):
        """Should return library statistics."""
        from backend.plex_client import PlexClient

        # Mock genre filter choices
        mock_genre1 = MagicMock()
        mock_genre1.title = "Rock"
        mock_genre2 = MagicMock()
        mock_genre2.title = "Alternative"

        # Mock decade filter choices
        mock_decade1 = MagicMock()
        mock_decade1.title = "1990"
        mock_decade2 = MagicMock()
        mock_decade2.title = "2000"

        mock_library = MagicMock()
        mock_library.totalViewSize.return_value = 10
        mock_library.listFilterChoices.side_effect = lambda field, libtype: (
            [mock_genre1, mock_genre2] if field == "genre" else [mock_decade1, mock_decade2]
        )

        mock_server = MagicMock()
        mock_server.library.section.return_value = mock_library

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://localhost:32400", "token", "Music")
            stats = client.get_library_stats()

            assert stats["total_tracks"] == 10
            assert len(stats["genres"]) == 2
            assert len(stats["decades"]) == 2


class TestPlexClientMusicLibraries:
    """Tests for music library listing."""

    def test_get_music_libraries(self, mocker):
        """Should return list of music libraries."""
        from backend.plex_client import PlexClient

        mock_section1 = MagicMock()
        mock_section1.title = "Music"
        mock_section1.type = "artist"

        mock_section2 = MagicMock()
        mock_section2.title = "Movies"
        mock_section2.type = "movie"

        mock_section3 = MagicMock()
        mock_section3.title = "Jazz Collection"
        mock_section3.type = "artist"

        mock_server = MagicMock()
        mock_server.library.sections.return_value = [mock_section1, mock_section2, mock_section3]
        mock_server.library.section.return_value = MagicMock()

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://localhost:32400", "token", "Music")
            libraries = client.get_music_libraries()

            assert "Music" in libraries
            assert "Jazz Collection" in libraries
            assert "Movies" not in libraries


class TestPlexClientTrackSearch:
    """Tests for track search functionality."""

    def test_search_tracks_by_title(self, mocker):
        """Should find tracks by title."""
        from backend.plex_client import PlexClient

        mock_track = MagicMock()
        mock_track.ratingKey = "123"
        mock_track.title = "Fake Plastic Trees"
        mock_track.grandparentTitle = "Radiohead"
        mock_track.parentTitle = "The Bends"
        mock_track.duration = 290000
        mock_track.parentYear = 1995
        mock_track.genres = [MagicMock(tag="Alternative")]

        mock_library = MagicMock()
        mock_library.searchTracks.return_value = [mock_track]
        mock_library.search.return_value = []

        mock_server = MagicMock()
        mock_server.library.section.return_value = mock_library

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://localhost:32400", "token", "Music")
            results = client.search_tracks("Fake Plastic")

            assert len(results) == 1
            assert results[0].title == "Fake Plastic Trees"
            assert results[0].artist == "Radiohead"

    def test_search_tracks_by_artist(self, mocker):
        """Should find tracks by artist name."""
        from backend.plex_client import PlexClient

        mock_track = MagicMock()
        mock_track.ratingKey = "456"
        mock_track.title = "Creep"
        mock_track.grandparentTitle = "Radiohead"
        mock_track.parentTitle = "Pablo Honey"
        mock_track.duration = 238000
        mock_track.parentYear = 1993
        mock_track.genres = []

        mock_library = MagicMock()
        mock_library.searchTracks.return_value = []
        mock_library.search.return_value = [mock_track]

        mock_server = MagicMock()
        mock_server.library.section.return_value = mock_library

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://localhost:32400", "token", "Music")
            results = client.search_tracks("Radiohead")

            assert len(results) == 1
            assert results[0].artist == "Radiohead"

    def test_search_tracks_returns_formatted_results(self, mocker):
        """Search results should include album art URLs."""
        from backend.plex_client import PlexClient

        mock_track = MagicMock()
        mock_track.ratingKey = "789"
        mock_track.title = "Black"
        mock_track.grandparentTitle = "Pearl Jam"
        mock_track.parentTitle = "Ten"
        mock_track.duration = 340000
        mock_track.parentYear = 1991
        mock_track.genres = [MagicMock(tag="Grunge")]

        mock_library = MagicMock()
        mock_library.searchTracks.return_value = [mock_track]
        mock_library.search.return_value = []

        mock_server = MagicMock()
        mock_server.library.section.return_value = mock_library

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://localhost:32400", "token", "Music")
            results = client.search_tracks("Black")

            assert len(results) == 1
            assert results[0].art_url == "/api/art/789"


class TestPlexClientPlaylistCreation:
    """Tests for playlist creation."""

    def test_create_playlist_success(self, mocker):
        """Should create playlist successfully."""
        from backend.plex_client import PlexClient

        mock_track = MagicMock()
        mock_playlist = MagicMock()
        mock_playlist.ratingKey = "999"

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.fetchItem.return_value = mock_track
        mock_server.createPlaylist.return_value = mock_playlist

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://localhost:32400", "token", "Music")
            result = client.create_playlist("Test Playlist", ["1", "2", "3"])

            assert result["success"] is True
            assert result["playlist_id"] == "999"

    def test_create_playlist_handles_invalid_tracks(self, mocker):
        """Should skip invalid track keys gracefully."""
        from backend.plex_client import PlexClient

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.fetchItem.side_effect = Exception("Not found")

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://localhost:32400", "token", "Music")
            result = client.create_playlist("Test Playlist", ["invalid"])

            assert result["success"] is False
            assert "error" in result

    def test_create_playlist_returns_playlist_url(self, mocker):
        """Should return playlist URL when successful."""
        from backend.plex_client import PlexClient

        mock_track = MagicMock()
        mock_playlist = MagicMock()
        mock_playlist.ratingKey = "999"

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.fetchItem.return_value = mock_track
        mock_server.createPlaylist.return_value = mock_playlist
        mock_server.machineIdentifier = "abc123def456"

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://localhost:32400", "token", "Music")
            result = client.create_playlist("Test Playlist", ["1", "2", "3"])

            assert result["success"] is True
            assert result["playlist_url"] is not None
            assert "abc123def456" in result["playlist_url"]
            assert "999" in result["playlist_url"]

    def test_playlist_url_uses_public_url(self, monkeypatch):
        """PLEX_PUBLIC_URL replaces PLEX_URL in 'Open in Plex' links."""
        from backend.plex_client import PlexClient

        monkeypatch.setenv("PLEX_PUBLIC_URL", "https://plex.example.com/")
        mock_playlist = MagicMock()
        mock_playlist.ratingKey = "999"

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.fetchItem.return_value = MagicMock()
        mock_server.createPlaylist.return_value = mock_playlist
        mock_server.machineIdentifier = "abc123def456"

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            client = PlexClient("http://plex:32400", "token", "Music")
            result = client.create_playlist("Test Playlist", ["1"])

        assert result["playlist_url"].startswith(
            "https://plex.example.com/web/index.html#!/server/abc123def456"
        )


class TestTrackCache:
    """Tests for track caching functionality."""

    def _make_track(self, rating_key: str, title: str = "Test Track") -> Track:
        """Helper to create a test track."""
        return Track(
            rating_key=rating_key,
            title=title,
            artist="Test Artist",
            album="Test Album",
            duration_ms=180000,
        )

    def test_cache_miss_returns_none(self):
        """Should return None for uncached filters."""
        cache = TrackCache()
        result = cache.get(["Rock"], ["1990s"], True, 0)
        assert result is None

    def test_cache_hit_returns_tracks(self):
        """Should return cached tracks for matching filters."""
        cache = TrackCache()
        tracks = [self._make_track("1"), self._make_track("2")]

        cache.set(["Rock"], ["1990s"], True, 0, tracks)
        result = cache.get(["Rock"], ["1990s"], True, 0)

        assert result is not None
        assert len(result) == 2
        assert result[0].rating_key == "1"

    def test_cache_expired_returns_none(self):
        """Should return None for expired entries."""
        cache = TrackCache(ttl_seconds=0)  # Immediate expiration
        tracks = [self._make_track("1")]

        cache.set(["Rock"], ["1990s"], True, 0, tracks)
        time.sleep(0.01)  # Ensure TTL expires
        result = cache.get(["Rock"], ["1990s"], True, 0)

        assert result is None

    def test_different_filters_are_separate_entries(self):
        """Should cache separately for different filter combinations."""
        cache = TrackCache()
        rock_tracks = [self._make_track("1", "Rock Song")]
        jazz_tracks = [self._make_track("2", "Jazz Song")]

        cache.set(["Rock"], [], True, 0, rock_tracks)
        cache.set(["Jazz"], [], True, 0, jazz_tracks)

        rock_result = cache.get(["Rock"], [], True, 0)
        jazz_result = cache.get(["Jazz"], [], True, 0)

        assert rock_result[0].title == "Rock Song"
        assert jazz_result[0].title == "Jazz Song"

    def test_key_generation_is_consistent(self):
        """Should generate same key regardless of genre/decade order."""
        cache = TrackCache()
        tracks = [self._make_track("1")]

        # Set with one order
        cache.set(["Rock", "Alternative"], ["1990s", "2000s"], True, 0, tracks)

        # Get with different order - should still hit
        result = cache.get(["Alternative", "Rock"], ["2000s", "1990s"], True, 0)

        assert result is not None

    def test_max_entries_evicts_oldest(self):
        """Should evict oldest entry when at capacity."""
        cache = TrackCache(max_entries=2)

        cache.set(["Rock"], [], True, 0, [self._make_track("1")])
        time.sleep(0.01)
        cache.set(["Jazz"], [], True, 0, [self._make_track("2")])
        time.sleep(0.01)
        cache.set(["Pop"], [], True, 0, [self._make_track("3")])  # Should evict Rock

        assert cache.get(["Rock"], [], True, 0) is None  # Evicted
        assert cache.get(["Jazz"], [], True, 0) is not None
        assert cache.get(["Pop"], [], True, 0) is not None

    def test_clear_removes_all_entries(self):
        """Should remove all entries on clear."""
        cache = TrackCache()
        cache.set(["Rock"], [], True, 0, [self._make_track("1")])
        cache.set(["Jazz"], [], True, 0, [self._make_track("2")])

        cache.clear()

        assert cache.get(["Rock"], [], True, 0) is None
        assert cache.get(["Jazz"], [], True, 0) is None

    def test_updating_existing_key_does_not_evict(self):
        """Should not evict when updating an existing entry."""
        cache = TrackCache(max_entries=2)

        cache.set(["Rock"], [], True, 0, [self._make_track("1")])
        cache.set(["Jazz"], [], True, 0, [self._make_track("2")])

        # Update Rock - should not trigger eviction
        cache.set(["Rock"], [], True, 0, [self._make_track("1-updated")])

        assert cache.get(["Rock"], [], True, 0) is not None
        assert cache.get(["Jazz"], [], True, 0) is not None


class TestPlexClientGetClients:
    """Tests for get_clients() method that discovers online Plex clients."""

    def _make_mock_client(
        self,
        machine_id: str = "abc123",
        title: str = "Living Room TV",
        product: str = "Plex for LG",
        platform: str = "webOS",
        protocol_capabilities: str = "timeline,playback,navigation",
        is_playing: bool = False,
    ) -> MagicMock:
        """Helper to create a mock Plex client resource."""
        client = MagicMock()
        client.machineIdentifier = machine_id
        client.title = title
        client.product = product
        client.platform = platform
        client.protocolCapabilities = protocol_capabilities
        client.isPlayingMedia.return_value = is_playing
        return client

    def test_get_clients_returns_playback_capable_clients(self):
        """Should only return clients that have 'playback' in protocolCapabilities."""
        from backend.plex_client import PlexClient

        playback_client = self._make_mock_client(
            machine_id="client1",
            title="Plexamp",
            protocol_capabilities="timeline,playback,navigation",
        )
        non_playback_client = self._make_mock_client(
            machine_id="client2",
            title="Plex Web",
            protocol_capabilities="timeline,navigation",
        )

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.clients.return_value = [playback_client, non_playback_client]

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            plex = PlexClient("http://localhost:32400", "token", "Music")
            clients = plex.get_clients()

        assert len(clients) == 1
        assert clients[0].client_id == "client1"
        assert clients[0].name == "Plexamp"

    def test_get_clients_detects_playing_state(self):
        """Should set is_playing=True when client is actively playing media."""
        from backend.plex_client import PlexClient

        playing_client = self._make_mock_client(
            machine_id="player1",
            title="Bedroom Speaker",
            is_playing=True,
        )

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.clients.return_value = [playing_client]

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            plex = PlexClient("http://localhost:32400", "token", "Music")
            clients = plex.get_clients()

        assert len(clients) == 1
        assert clients[0].is_playing is True

    def test_get_clients_detects_idle_state(self):
        """Should set is_playing=False when client is idle."""
        from backend.plex_client import PlexClient

        idle_client = self._make_mock_client(
            machine_id="idle1",
            title="Kitchen Speaker",
            is_playing=False,
        )

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.clients.return_value = [idle_client]

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            plex = PlexClient("http://localhost:32400", "token", "Music")
            clients = plex.get_clients()

        assert len(clients) == 1
        assert clients[0].is_playing is False

    def test_get_clients_skips_unresponsive_clients(self):
        """Should exclude clients where isPlayingMedia() raises an exception."""
        from backend.plex_client import PlexClient

        responsive_client = self._make_mock_client(
            machine_id="good1",
            title="Working Player",
        )
        unresponsive_client = self._make_mock_client(
            machine_id="bad1",
            title="Broken Player",
        )
        unresponsive_client.isPlayingMedia.side_effect = Exception(
            "Connection timed out"
        )

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.clients.return_value = [responsive_client, unresponsive_client]

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            plex = PlexClient("http://localhost:32400", "token", "Music")
            clients = plex.get_clients()

        # Only the responsive client should be returned
        assert len(clients) == 1
        assert clients[0].client_id == "good1"
        assert clients[0].name == "Working Player"

    def test_get_clients_maps_correct_fields(self):
        """Should map machineIdentifier->client_id, title->name, product, platform."""
        from backend.plex_client import PlexClient
        from backend.models import PlexClientInfo

        client = self._make_mock_client(
            machine_id="xyz789",
            title="Eric's iPhone",
            product="Plexamp",
            platform="iOS",
            is_playing=False,
        )

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.clients.return_value = [client]

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            plex = PlexClient("http://localhost:32400", "token", "Music")
            clients = plex.get_clients()

        assert len(clients) == 1
        result = clients[0]
        assert isinstance(result, PlexClientInfo)
        assert result.client_id == "xyz789"
        assert result.name == "Eric's iPhone"
        assert result.product == "Plexamp"
        assert result.platform == "iOS"
        assert result.is_playing is False


class TestPlexClientPlayQueue:
    """Tests for play_queue() method that sends tracks to a Plex client."""

    def _make_mock_client(
        self,
        machine_id: str = "abc123",
        title: str = "Living Room TV",
        product: str = "Plexamp",
        protocol_capabilities: str = "timeline,playback,navigation",
    ) -> MagicMock:
        """Helper to create a mock Plex client resource."""
        client = MagicMock()
        client.machineIdentifier = machine_id
        client.title = title
        client.product = product
        client.protocolCapabilities = protocol_capabilities
        return client

    def test_play_queue_replace_mode(self):
        """Should create a PlayQueue and send it to the client in replace mode."""
        from backend.plex_client import PlexClient

        # Set up mock tracks returned by server.fetchItem()
        mock_track1 = MagicMock()
        mock_track1.ratingKey = 101
        mock_track2 = MagicMock()
        mock_track2.ratingKey = 102

        # Set up mock client
        target_client = self._make_mock_client(
            machine_id="target1",
            title="Plexamp Mobile",
            product="Plexamp",
        )

        mock_play_queue = MagicMock()

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.clients.return_value = [target_client]
        mock_server.fetchItem.side_effect = lambda key: {
            101: mock_track1,
            102: mock_track2,
        }[key]

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            with patch("backend.plex_client.PlayQueue") as MockPlayQueue:
                MockPlayQueue.create.return_value = mock_play_queue

                plex = PlexClient("http://localhost:32400", "token", "Music")
                result = plex.play_queue(
                    rating_keys=["101", "102"],
                    client_id="target1",
                    mode="replace",
                )

        # Verify proxyThroughServer was called
        target_client.proxyThroughServer.assert_called_once()

        # Verify PlayQueue.create was called with correct arguments
        MockPlayQueue.create.assert_called_once()
        create_call = MockPlayQueue.create.call_args
        # Check items contain both tracks
        assert mock_track1 in create_call.kwargs.get("items", create_call[1].get("items", []))
        assert mock_track2 in create_call.kwargs.get("items", create_call[1].get("items", []))
        # Check startItem is the first track
        start_item = create_call.kwargs.get("startItem", create_call[1].get("startItem"))
        assert start_item == mock_track1

        # Verify playMedia was called with the play queue
        target_client.playMedia.assert_called_once_with(mock_play_queue)

        # Verify return values
        assert result["success"] is True
        assert result["client_name"] == "Plexamp Mobile"
        assert result["client_product"] == "Plexamp"
        assert result["tracks_queued"] == 2

    def test_play_queue_hands_player_public_url(self, monkeypatch):
        """With PLEX_PUBLIC_URL set, the player is told to fetch from it, not PLEX_URL."""
        from backend.plex_client import PlexClient

        monkeypatch.setenv("PLEX_PUBLIC_URL", "https://plex.example.com")
        target_client = self._make_mock_client(machine_id="target1")
        mock_play_queue = MagicMock()

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.clients.return_value = [target_client]
        mock_server.fetchItem.return_value = MagicMock()

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            with patch("backend.plex_client.PlayQueue") as MockPlayQueue:
                MockPlayQueue.create.return_value = mock_play_queue

                plex = PlexClient("http://plex:32400", "token", "Music")
                result = plex.play_queue(rating_keys=["101"], client_id="target1", mode="replace")

        assert result["success"] is True
        target_client.playMedia.assert_called_once_with(
            mock_play_queue, protocol="https", address="plex.example.com", port="443"
        )

    class FakeQueueItem:
        def __init__(self, rating_key, item_id):
            self.ratingKey = rating_key
            self.playQueueItemID = item_id

    class FakePlayQueue:
        """Behaves like a Plex play queue: add appends, moveItem reorders."""

        def __init__(self, rating_keys, current_index=0):
            self.items = [TestPlexClientPlayQueue.FakeQueueItem(k, i) for i, k in enumerate(rating_keys)]
            self.selectedItem = self.items[current_index]
            self._next_id = len(self.items)

        def addItem(self, track, playNext=False, refresh=True):
            self.items.append(TestPlexClientPlayQueue.FakeQueueItem(track.ratingKey, self._next_id))
            self._next_id += 1

        def refresh(self):
            pass

        def moveItem(self, item, after=None, refresh=True):
            self.items.remove(item)
            self.items.insert(self.items.index(after) + 1 if after else 0, item)

        def order(self):
            return [item.ratingKey for item in self.items]

    def _play_next(self, queue, rating_keys):
        from backend.plex_client import PlexClient

        tracks = {int(k): MagicMock(ratingKey=int(k)) for k in rating_keys}
        target_client = self._make_mock_client(machine_id="target2", title="Desktop Player", product="Plexamp")
        target_client.timelines.return_value = [MagicMock(type="music", playQueueID=42)]
        mock_server = MagicMock()
        mock_server.clients.return_value = [target_client]
        mock_server.fetchItem.side_effect = lambda key: tracks[key]

        with patch("backend.plex_client.PlexServer", return_value=mock_server), \
             patch("backend.plex_client.PlayQueue") as MockPlayQueue:
            MockPlayQueue.get.return_value = queue
            plex = PlexClient("http://localhost:32400", "token", "Music")
            result = plex.play_queue(rating_keys=rating_keys, client_id="target2", mode="play_next")
        return result, target_client, MockPlayQueue

    def test_play_queue_play_next_mode(self):
        """New tracks go right after the current one, in order, ahead of the existing up next (#27)."""
        queue = self.FakePlayQueue([100, 101, 102, 103], current_index=1)

        result, target_client, MockPlayQueue = self._play_next(queue, ["201", "202", "203"])

        assert queue.order() == [100, 101, 201, 202, 203, 102, 103]
        assert MockPlayQueue.get.call_args.args[1] == 42
        target_client.proxyThroughServer.assert_called_once()
        target_client.refreshPlayQueue.assert_called_once_with(42, mtype="music")
        assert result["success"] is True
        assert result["tracks_queued"] == 3
        assert result["client_name"] == "Desktop Player"

    def test_play_next_with_track_already_in_queue(self):
        """A track already further down the queue is queued again next, not moved from there."""
        queue = self.FakePlayQueue([100, 101, 202], current_index=0)

        self._play_next(queue, ["202", "203"])

        assert queue.order() == [100, 202, 203, 101, 202]

    def test_play_queue_offline_client(self):
        """Should return success=False with error when client is offline."""
        from backend.plex_client import PlexClient

        mock_track = MagicMock()
        mock_track.ratingKey = 301

        target_client = self._make_mock_client(
            machine_id="offline1",
            title="Offline Player",
            product="Plexamp",
        )
        # Simulate offline client - playMedia raises requests ConnectionError
        target_client.playMedia.side_effect = RequestsConnectionError("Connection refused")

        mock_play_queue = MagicMock()

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.clients.return_value = [target_client]
        mock_server.fetchItem.return_value = mock_track

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            with patch("backend.plex_client.PlayQueue") as MockPlayQueue:
                MockPlayQueue.create.return_value = mock_play_queue

                plex = PlexClient("http://localhost:32400", "token", "Music")
                result = plex.play_queue(
                    rating_keys=["301"],
                    client_id="offline1",
                    mode="replace",
                )

        assert result["success"] is False
        assert "went offline" in result["error"]

    def test_play_queue_client_not_found(self):
        """Should return error when target client_id is not among connected clients."""
        from backend.plex_client import PlexClient

        # Create a client with a different machineIdentifier than what we'll request
        other_client = self._make_mock_client(
            machine_id="other_machine",
            title="Other Player",
        )

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.clients.return_value = [other_client]

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            plex = PlexClient("http://localhost:32400", "token", "Music")
            result = plex.play_queue(
                rating_keys=["100"],
                client_id="nonexistent_id",
                mode="replace",
            )

        assert result["success"] is False
        assert result["error"] is not None


class TestPlexClientGetPlaylists:
    """Tests for get_playlists() method that returns audio playlists."""

    def _make_mock_playlist(self, rating_key=1, title="Test Playlist", leaf_count=10, smart=False, radio=False):
        """Helper to create a mock Plex playlist object."""
        playlist = MagicMock()
        playlist.ratingKey = rating_key
        playlist.title = title
        playlist.leafCount = leaf_count
        playlist.smart = smart
        playlist.radio = radio
        return playlist

    def test_get_playlists_returns_audio_playlists(self):
        """Should return audio playlists as PlexPlaylistInfo objects."""
        from backend.plex_client import PlexClient

        pl1 = self._make_mock_playlist(rating_key=100, title="Road Trip Mix", leaf_count=25)
        pl2 = self._make_mock_playlist(rating_key=200, title="Chill Vibes", leaf_count=18)

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.playlists.return_value = [pl1, pl2]

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            plex = PlexClient("http://localhost:32400", "token", "Music")
            playlists = plex.get_playlists()

        # Verify server.playlists was called for audio type
        mock_server.playlists.assert_called_once_with(playlistType="audio")

        assert len(playlists) == 2
        # Check types and field mapping
        for p in playlists:
            assert isinstance(p, PlexPlaylistInfo)

        # Sorted alphabetically: Chill Vibes, Road Trip Mix
        assert playlists[0].rating_key == "200"
        assert playlists[0].title == "Chill Vibes"
        assert playlists[0].track_count == 18
        assert playlists[1].rating_key == "100"
        assert playlists[1].title == "Road Trip Mix"
        assert playlists[1].track_count == 25

    def test_get_playlists_sorted_by_title(self):
        """Should return playlists sorted alphabetically by title."""
        from backend.plex_client import PlexClient

        pl_z = self._make_mock_playlist(rating_key=1, title="Zen Garden", leaf_count=5)
        pl_a = self._make_mock_playlist(rating_key=2, title="Afternoon Jazz", leaf_count=12)
        pl_m = self._make_mock_playlist(rating_key=3, title="Morning Run", leaf_count=20)

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.playlists.return_value = [pl_z, pl_a, pl_m]

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            plex = PlexClient("http://localhost:32400", "token", "Music")
            playlists = plex.get_playlists()

        titles = [p.title for p in playlists]
        assert titles == ["Afternoon Jazz", "Morning Run", "Zen Garden"]

    def test_get_playlists_empty(self):
        """Should return empty list when server has no audio playlists."""
        from backend.plex_client import PlexClient

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.playlists.return_value = []

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            plex = PlexClient("http://localhost:32400", "token", "Music")
            playlists = plex.get_playlists()

        assert playlists == []

    def test_get_playlists_excludes_smart_and_radio(self):
        """Should exclude smart playlists and radio playlists from results."""
        from backend.plex_client import PlexClient

        regular = self._make_mock_playlist(rating_key=1, title="My Mix", leaf_count=10)
        smart = self._make_mock_playlist(rating_key=2, title="Favorite Songs", leaf_count=50, smart=True)
        radio = self._make_mock_playlist(rating_key=3, title="Radio Station", leaf_count=0, radio=True)

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.playlists.return_value = [regular, smart, radio]

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            plex = PlexClient("http://localhost:32400", "token", "Music")
            playlists = plex.get_playlists()

        assert len(playlists) == 1
        assert playlists[0].title == "My Mix"


class TestPlexClientUpdatePlaylist:
    """Tests for update_playlist() method that modifies existing playlists."""

    def _make_mock_playlist(self, rating_key=1, title="Test Playlist", leaf_count=10, items=None):
        """Helper to create a mock Plex playlist object."""
        playlist = MagicMock()
        playlist.ratingKey = rating_key
        playlist.title = title
        playlist.leafCount = leaf_count
        playlist.items.return_value = items or []
        return playlist

    def _make_mock_track(self, rating_key):
        """Helper to create a mock Plex track object."""
        track = MagicMock()
        track.ratingKey = int(rating_key)
        return track

    def test_update_playlist_replace_mode(self):
        """Should remove existing items and add new ones in replace mode."""
        from backend.plex_client import PlexClient

        # Existing tracks in playlist
        existing_track_a = self._make_mock_track(50)
        existing_track_b = self._make_mock_track(51)

        # The playlist we'll fetch
        mock_playlist = self._make_mock_playlist(
            rating_key=10,
            title="My Playlist",
            leaf_count=2,
            items=[existing_track_a, existing_track_b],
        )

        # New tracks to add
        new_track_1 = self._make_mock_track(101)
        new_track_2 = self._make_mock_track(102)
        new_track_3 = self._make_mock_track(103)

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.fetchItem.side_effect = lambda key: {
            10: mock_playlist,
            101: new_track_1,
            102: new_track_2,
            103: new_track_3,
        }[key]
        mock_server.machineIdentifier = "server123"

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            plex = PlexClient("http://localhost:32400", "token", "Music")
            result = plex.update_playlist(
                playlist_id="10",
                rating_keys=["101", "102", "103"],
                mode="replace",
            )

        # Verify existing items were removed
        mock_playlist.removeItems.assert_called_once_with([existing_track_a, existing_track_b])
        # Verify new items were added
        mock_playlist.addItems.assert_called_once_with([new_track_1, new_track_2, new_track_3])

        assert result["success"] is True
        assert result["tracks_added"] == 3

    def test_update_playlist_append_mode_deduplicates(self):
        """Should skip tracks that already exist in the playlist when appending."""
        from backend.plex_client import PlexClient

        # Existing tracks: 101 and 102 already in playlist
        existing_track_101 = self._make_mock_track(101)
        existing_track_102 = self._make_mock_track(102)

        mock_playlist = self._make_mock_playlist(
            rating_key=20,
            title="Append Test",
            leaf_count=2,
            items=[existing_track_101, existing_track_102],
        )

        # New tracks to add: 101 (dup), 103 (new), 102 (dup), 104 (new)
        new_track_101 = self._make_mock_track(101)
        new_track_102 = self._make_mock_track(102)
        new_track_103 = self._make_mock_track(103)
        new_track_104 = self._make_mock_track(104)

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.fetchItem.side_effect = lambda key: {
            20: mock_playlist,
            101: new_track_101,
            102: new_track_102,
            103: new_track_103,
            104: new_track_104,
        }[key]
        mock_server.machineIdentifier = "server123"

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            plex = PlexClient("http://localhost:32400", "token", "Music")
            result = plex.update_playlist(
                playlist_id="20",
                rating_keys=["101", "103", "102", "104"],
                mode="append",
            )

        # Only non-duplicate tracks (103, 104) should be added
        mock_playlist.addItems.assert_called_once()
        added_items = mock_playlist.addItems.call_args[0][0]
        added_keys = [t.ratingKey for t in added_items]
        assert 103 in added_keys
        assert 104 in added_keys
        assert 101 not in added_keys
        assert 102 not in added_keys

        assert result["success"] is True
        assert result["tracks_added"] == 2
        assert result["duplicates_skipped"] == 2

    def test_update_playlist_scratch_creates_if_not_found(self):
        """Should create 'MediaSage - Now Playing' when __scratch__ and no existing match."""
        from backend.plex_client import PlexClient

        new_track_1 = self._make_mock_track(201)
        new_track_2 = self._make_mock_track(202)

        created_playlist = self._make_mock_playlist(
            rating_key=999,
            title="MediaSage - Now Playing",
            leaf_count=2,
        )

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        # No existing "MediaSage - Now Playing" playlist
        mock_server.playlists.return_value = []
        mock_server.createPlaylist.return_value = created_playlist
        mock_server.fetchItem.side_effect = lambda key: {
            201: new_track_1,
            202: new_track_2,
        }[key]
        mock_server.machineIdentifier = "server123"

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            plex = PlexClient("http://localhost:32400", "token", "Music")
            result = plex.update_playlist(
                playlist_id="__scratch__",
                rating_keys=["201", "202"],
                mode="replace",
            )

        # Verify createPlaylist was called with correct title and track items
        mock_server.createPlaylist.assert_called_once()
        create_call = mock_server.createPlaylist.call_args
        # Title should be "MediaSage - Now Playing" — check positional or keyword arg
        call_title = create_call[0][0] if create_call[0] else create_call[1].get("title")
        assert call_title == "MediaSage - Now Playing"

        assert result["success"] is True

    def test_update_playlist_scratch_uses_existing(self):
        """Should reuse existing 'MediaSage - Now Playing' playlist for __scratch__."""
        from backend.plex_client import PlexClient

        existing_scratch = self._make_mock_playlist(
            rating_key=500,
            title="MediaSage - Now Playing",
            leaf_count=5,
            items=[self._make_mock_track(1), self._make_mock_track(2)],
        )

        new_track = self._make_mock_track(301)

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        # Return existing scratch playlist
        mock_server.playlists.return_value = [existing_scratch]
        mock_server.fetchItem.side_effect = lambda key: {
            500: existing_scratch,
            301: new_track,
        }[key]
        mock_server.machineIdentifier = "server123"

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            plex = PlexClient("http://localhost:32400", "token", "Music")
            result = plex.update_playlist(
                playlist_id="__scratch__",
                rating_keys=["301"],
                mode="replace",
            )

        # Should NOT create a new playlist
        mock_server.createPlaylist.assert_not_called()
        # Should use the existing playlist (fetch it by ratingKey 500)
        assert result["success"] is True

    def test_update_playlist_tracks_skipped_on_fetch_failure(self):
        """Should count tracks that fail to fetch and report them as skipped."""
        from backend.plex_client import PlexClient

        good_track = self._make_mock_track(401)
        mock_playlist = self._make_mock_playlist(
            rating_key=30,
            title="Skip Test",
            leaf_count=0,
            items=[],
        )

        def fetch_item_side_effect(key):
            if key == 30:
                return mock_playlist
            if key == 401:
                return good_track
            raise Exception(f"Track {key} not found")

        mock_server = MagicMock()
        mock_server.library.section.return_value = MagicMock()
        mock_server.fetchItem.side_effect = fetch_item_side_effect
        mock_server.machineIdentifier = "server123"

        with patch("backend.plex_client.PlexServer", return_value=mock_server):
            plex = PlexClient("http://localhost:32400", "token", "Music")
            result = plex.update_playlist(
                playlist_id="30",
                rating_keys=["401", "402", "403"],
                mode="replace",
            )

        assert result["success"] is True
        assert result["tracks_added"] == 1
        assert result["tracks_skipped"] == 2
