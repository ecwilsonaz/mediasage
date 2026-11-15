"""Jellyfin media server client using the REST API."""

import logging
import re
from collections.abc import Iterator
from typing import Any

import httpx

from backend.media_client import BaseMediaClient, is_live_track
from backend.models import PlexPlaylistInfo, Track

logger = logging.getLogger(__name__)

_jellyfin_client: "JellyfinClient | None" = None

_ITEM_ID_RE = re.compile(r"^[0-9a-f]{32}$")
TRACK_FIELDS = "Genres,ProductionYear,RunTimeTicks,AlbumArtist,Album,Artists"
PAGE_SIZE = 1000
SCRATCH_PLAYLIST_TITLE = "MediaSage - Now Playing"


def _years_param(decades: list[str]) -> str | None:
    """Turn ["1980s", "1990s"] into Jellyfin's comma-separated Years filter."""
    years = []
    for decade in decades:
        try:
            start = int(decade.strip().rstrip("s"))
        except ValueError:
            continue
        years.extend(range(start, start + 10))
    return ",".join(map(str, years)) or None


class JellyfinClient(BaseMediaClient):
    """Client for a Jellyfin server's music library."""

    def __init__(self, url: str, token: str, music_library: str = "Music"):
        self.url = url.rstrip("/")
        self.music_library_name = music_library

        self._connected = False
        self._error: str | None = None
        self._user_id: str | None = None
        self._library_id: str | None = None

        self._http = httpx.Client(
            base_url=self.url,
            headers={
                "Authorization": (
                    f'MediaBrowser Client="MediaSage", Device="Server", '
                    f'DeviceId="mediasage-server", Version="1.0.0", Token="{token}"'
                ),
            },
            timeout=30.0,
        )

        if not url or not token:
            self._error = "Jellyfin URL and token are required"
        else:
            self._connect()

    def _connect(self) -> None:
        try:
            # API keys aren't tied to a user (/Users/Me returns 400), so act as the first admin
            resp = self._http.get("/Users")
            resp.raise_for_status()
            users = resp.json()
            if not users:
                self._error = "No users found in Jellyfin"
                return
            admin = next((u for u in users if u.get("Policy", {}).get("IsAdministrator")), users[0])
            self._user_id = admin["Id"]

            resp = self._http.get("/Library/MediaFolders")
            resp.raise_for_status()
            folders = resp.json().get("Items", [])
            wanted = self.music_library_name.lower()
            library = next((f for f in folders if f.get("Name", "").lower() == wanted), None)
            library = library or next((f for f in folders if f.get("CollectionType") == "music"), None)
            if not library:
                self._error = f"Music library '{self.music_library_name}' not found in Jellyfin"
                return

            self._library_id = library["Id"]
            self._connected = True
            self._error = None
            logger.info("Connected to Jellyfin: user_id=%s library_id=%s", self._user_id, self._library_id)
        except httpx.ConnectError:
            self._error = f"Cannot connect to Jellyfin server at {self.url}"
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                self._error = "Invalid Jellyfin API key — unauthorized"
            else:
                self._error = f"Jellyfin returned HTTP {e.response.status_code}"
        except Exception as e:
            self._error = f"Jellyfin connection error: {e}"

    def is_connected(self) -> bool:
        return self._connected

    def get_error(self) -> str | None:
        return self._error

    def _library_params(self, item_type: str = "Audio", **extra: Any) -> dict[str, Any]:
        return {
            "IncludeItemTypes": item_type,
            "Recursive": "true",
            "ParentId": self._library_id,
            "UserId": self._user_id,
            **extra,
        }

    def _filter_params(
        self, genres: list[str] | None, decades: list[str] | None, **extra: Any
    ) -> dict[str, Any]:
        params = self._library_params(**extra)
        if genres:
            params["Genres"] = "|".join(genres)
        if decades and (years := _years_param(decades)):
            params["Years"] = years
        return params

    def _iter_items(self, params: dict[str, Any]) -> Iterator[dict]:
        """Page through /Items. Raises on HTTP errors."""
        start_index = 0
        while True:
            resp = self._http.get(
                "/Items",
                params={**params, "StartIndex": start_index, "Limit": PAGE_SIZE},
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("Items", [])
            yield from items
            start_index += len(items)
            if not items or start_index >= data.get("TotalRecordCount", 0):
                break

    def _get_item(self, item_id: str, **params: Any) -> dict | None:
        # /Items/{id} needs a user context, so look the item up by ID instead
        try:
            resp = self._http.get("/Items", params={"Ids": item_id, "UserId": self._user_id, **params})
            resp.raise_for_status()
        except Exception as e:
            logger.warning("Failed to get Jellyfin item %s: %s", item_id, e)
            return None
        items = resp.json().get("Items", [])
        return items[0] if items else None

    def _item_to_track(self, item: dict) -> Track:
        return Track(
            rating_key=item["Id"],
            title=item.get("Name", ""),
            artist=item.get("AlbumArtist") or (item.get("Artists") or [""])[0],
            album=item.get("Album", ""),
            duration_ms=(item.get("RunTimeTicks") or 0) // 10_000,
            year=item.get("ProductionYear"),
            genres=item.get("Genres", []),
            art_url=f"/api/art/{item['Id']}",
        )

    def get_music_libraries(self) -> list[str]:
        try:
            resp = self._http.get("/Library/MediaFolders", timeout=10.0)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("Failed to get Jellyfin music libraries: %s", e)
            return []
        return [f["Name"] for f in resp.json().get("Items", []) if f.get("CollectionType") == "music"]

    def get_library_stats(self) -> dict[str, Any]:
        """Count tracks, genres and decades from the tracks themselves.

        Jellyfin's /Genres index can be empty for music libraries.
        """
        if not self._connected:
            return {"total_tracks": 0, "genres": [], "decades": []}

        genre_counts: dict[str, int] = {}
        decade_counts: dict[str, int] = {}
        total_tracks = 0
        try:
            for item in self._iter_items(self._library_params(Fields="Genres,ProductionYear")):
                total_tracks += 1
                for genre in item.get("Genres") or []:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
                if year := item.get("ProductionYear"):
                    decade = f"{year // 10 * 10}s"
                    decade_counts[decade] = decade_counts.get(decade, 0) + 1
        except Exception as e:
            logger.exception("Failed to get Jellyfin library stats")
            return {"total_tracks": 0, "genres": [], "decades": [], "error": str(e)}

        return {
            "total_tracks": total_tracks,
            "genres": [{"name": g, "count": c} for g, c in sorted(genre_counts.items())],
            "decades": [{"name": d, "count": c} for d, c in sorted(decade_counts.items())],
        }

    def get_all_tracks(self) -> list[Track]:
        """Get every track in the library. Raises on failure."""
        return [row["track"] for row in self.get_all_tracks_for_sync()]

    def get_all_tracks_for_sync(self) -> list[dict[str, Any]]:
        """Get every track with the album ID and play history the cache stores.

        Raises on failure, so an interrupted sync keeps the previous cache.

        Returns:
            Dicts with track (Track), album_key, view_count and last_viewed_at
        """
        if not self._connected:
            raise RuntimeError(self._error or "Not connected to Jellyfin")

        rows = []
        for item in self._iter_items(self._library_params(Fields=TRACK_FIELDS)):
            user_data = item.get("UserData") or {}
            rows.append({
                "track": self._item_to_track(item),
                "album_key": item.get("AlbumId") or "",
                "view_count": user_data.get("PlayCount") or 0,
                "last_viewed_at": user_data.get("LastPlayedDate"),
            })
        return rows

    def get_tracks_by_filters(
        self,
        genres: list[str] | None = None,
        decades: list[str] | None = None,
        exclude_live: bool = True,
        min_rating: int = 0,
        limit: int = 0,
    ) -> list[Track]:
        """Get tracks matching the filters. Jellyfin has no star ratings, so min_rating is ignored."""
        if not self._connected:
            return []

        tracks = []
        try:
            for item in self._iter_items(self._filter_params(genres, decades, Fields=TRACK_FIELDS)):
                track = self._item_to_track(item)
                if exclude_live and is_live_track(track.title, track.album):
                    continue
                tracks.append(track)
                if limit and len(tracks) >= limit:
                    break
        except Exception:
            logger.exception("Failed to get Jellyfin filtered tracks")
            return []
        return tracks

    def get_random_tracks(self, count: int, exclude_live: bool = True) -> list[Track]:
        if not self._connected:
            return []
        # Over-fetch so enough are left after dropping live tracks
        params = self._library_params(
            Fields=TRACK_FIELDS, SortBy="Random", Limit=count * 3 if exclude_live else count
        )
        try:
            resp = self._http.get("/Items", params=params)
            resp.raise_for_status()
        except Exception:
            logger.exception("Failed to get random Jellyfin tracks")
            return []

        tracks = [self._item_to_track(item) for item in resp.json().get("Items", [])]
        if exclude_live:
            tracks = [t for t in tracks if not is_live_track(t.title, t.album)]
        return tracks[:count]

    def get_track_by_key(self, rating_key: str) -> Track | None:
        if not self._connected or not _ITEM_ID_RE.match(rating_key):
            return None
        item = self._get_item(rating_key, Fields=TRACK_FIELDS)
        return self._item_to_track(item) if item else None

    def search_tracks(self, query: str) -> list[Track]:
        if not self._connected:
            return []
        params = self._library_params(SearchTerm=query, Fields=TRACK_FIELDS, Limit=50)
        try:
            resp = self._http.get("/Items", params=params)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("Failed to search Jellyfin tracks: %s", e)
            return []
        return [self._item_to_track(item) for item in resp.json().get("Items", [])]

    def count_tracks_by_filters(
        self,
        genres: list[str] | None = None,
        decades: list[str] | None = None,
        exclude_live: bool = True,
        min_rating: int = 0,
    ) -> int:
        if not self._connected:
            return 0
        # Jellyfin can't filter out live tracks, so counting them needs the full list
        if exclude_live:
            return len(self.get_tracks_by_filters(genres, decades, exclude_live=True))
        try:
            resp = self._http.get("/Items", params=self._filter_params(genres, decades, Limit=0))
            resp.raise_for_status()
        except Exception as e:
            logger.warning("Failed to count Jellyfin tracks: %s", e)
            return 0
        return resp.json().get("TotalRecordCount", 0)

    def create_playlist(self, name: str, rating_keys: list[str], description: str = "") -> dict[str, Any]:
        """Create a playlist. Jellyfin playlists have no description, so it's ignored."""
        if not self._connected:
            return {"success": False, "error": "Not connected to Jellyfin"}
        try:
            resp = self._http.post(
                "/Playlists",
                json={"Name": name, "Ids": rating_keys, "UserId": self._user_id, "MediaType": "Audio"},
            )
            resp.raise_for_status()
        except Exception as e:
            logger.exception("Failed to create Jellyfin playlist '%s'", name)
            return {"success": False, "error": str(e)}

        playlist_id = resp.json().get("Id")
        if not playlist_id:
            return {"success": False, "error": "Playlist created but no ID returned"}
        return {
            "success": True,
            "playlist_id": playlist_id,
            "playlist_url": None,
            "tracks_added": len(rating_keys),
            "tracks_skipped": 0,
        }

    def update_playlist(
        self,
        playlist_id: str,
        rating_keys: list[str],
        mode: str = "replace",
        description: str = "",
    ) -> dict[str, Any]:
        """Replace or append a playlist's tracks.

        Handles the __scratch__ sentinel for "MediaSage - Now Playing", like PlexClient.
        """
        if not self._connected:
            return {"success": False, "error": "Not connected to Jellyfin"}
        if mode not in ("replace", "append"):
            return {"success": False, "error": f"Unknown update mode: {mode}"}

        try:
            if playlist_id == "__scratch__":
                playlist_id = self._find_playlist_id(SCRATCH_PLAYLIST_TITLE)
                if not playlist_id:
                    created = self.create_playlist(SCRATCH_PLAYLIST_TITLE, rating_keys)
                    if not created["success"]:
                        return created
                    return {
                        "success": True,
                        "tracks_added": created["tracks_added"],
                        "tracks_skipped": 0,
                        "duplicates_skipped": 0,
                        "playlist_url": None,
                    }
            elif not _ITEM_ID_RE.match(playlist_id):
                return {"success": False, "error": "Invalid playlist ID"}

            entries = self._get_playlist_entries(playlist_id)
            if mode == "append":
                existing = {item_id for _, item_id in entries}
                to_add = [key for key in rating_keys if key not in existing]
            else:
                to_add = rating_keys
                # Jellyfin won't add an item that's already in the playlist, so clear it first
                if entries:
                    self._remove_playlist_entries(playlist_id, [entry_id for entry_id, _ in entries])

            if to_add:
                try:
                    self._add_playlist_items(playlist_id, to_add)
                except Exception:
                    if mode == "replace" and entries:
                        logger.warning("Adding tracks failed; restoring playlist %s", playlist_id)
                        self._add_playlist_items(playlist_id, [item_id for _, item_id in entries])
                    raise
        except Exception as e:
            logger.exception("Failed to update Jellyfin playlist %s", playlist_id)
            return {"success": False, "error": str(e)}

        return {
            "success": True,
            "tracks_added": len(to_add),
            "tracks_skipped": 0,
            "duplicates_skipped": len(rating_keys) - len(to_add),
            "playlist_url": None,
        }

    def _find_playlist_id(self, title: str) -> str | None:
        params = {"IncludeItemTypes": "Playlist", "Recursive": "true", "UserId": self._user_id, "SearchTerm": title}
        resp = self._http.get("/Items", params=params)
        resp.raise_for_status()
        return next((item["Id"] for item in resp.json().get("Items", []) if item.get("Name") == title), None)

    def _get_playlist_entries(self, playlist_id: str) -> list[tuple[str, str]]:
        """Return (playlist entry ID, item ID) pairs."""
        resp = self._http.get(f"/Playlists/{playlist_id}/Items", params={"UserId": self._user_id})
        resp.raise_for_status()
        return [
            (item["PlaylistItemId"], item["Id"])
            for item in resp.json().get("Items", [])
            if "PlaylistItemId" in item
        ]

    def _remove_playlist_entries(self, playlist_id: str, entry_ids: list[str]) -> None:
        resp = self._http.delete(f"/Playlists/{playlist_id}/Items", params={"EntryIds": ",".join(entry_ids)})
        resp.raise_for_status()

    def _add_playlist_items(self, playlist_id: str, item_ids: list[str]) -> None:
        params = {"Ids": ",".join(item_ids), "UserId": self._user_id}
        resp = self._http.post(f"/Playlists/{playlist_id}/Items", params=params)
        resp.raise_for_status()

    def get_playlists(self) -> list[PlexPlaylistInfo]:
        if not self._connected:
            return []
        params = {"IncludeItemTypes": "Playlist", "Recursive": "true", "UserId": self._user_id, "Fields": "ChildCount"}
        try:
            resp = self._http.get("/Items", params=params)
            resp.raise_for_status()
        except Exception:
            logger.exception("Failed to get Jellyfin playlists")
            return []

        playlists = [
            PlexPlaylistInfo(rating_key=item["Id"], title=item["Name"], track_count=item.get("ChildCount", 0))
            for item in resp.json().get("Items", [])
            # Imported .m3u playlists report MediaType "Unknown"; skip only video ones
            if (item.get("MediaType") or "").lower() != "video"
        ]
        return sorted(playlists, key=lambda p: p.title.lower())

    def get_art_url(self, item_id: str) -> str | None:
        """Get the image URL for a track or album, which main.py proxies.

        Falls back to the album's image, since most tracks have none of their own.
        """
        if not self._connected or not _ITEM_ID_RE.match(item_id):
            return None
        item = self._get_item(item_id)
        if not item:
            return None
        if (item.get("ImageTags") or {}).get("Primary"):
            return f"{self.url}/Items/{item_id}/Images/Primary"
        if item.get("AlbumId") and item.get("AlbumPrimaryImageTag"):
            return f"{self.url}/Items/{item['AlbumId']}/Images/Primary"
        return None

    def _system_info(self) -> dict:
        try:
            resp = self._http.get("/System/Info", timeout=10.0)
            resp.raise_for_status()
        except Exception:
            return {}
        return resp.json()

    def get_machine_identifier(self) -> str | None:
        """Server ID, used to detect a server change for the library cache."""
        return self._system_info().get("Id")

    def get_library_identifier(self) -> str | None:
        """Music library ID, used to detect a change of music library for the library cache."""
        return self._library_id

    def get_server_name(self) -> str | None:
        return self._system_info().get("ServerName")

    def close(self) -> None:
        self._http.close()


def get_jellyfin_client() -> JellyfinClient | None:
    return _jellyfin_client


def init_jellyfin_client(url: str, token: str, music_library: str = "Music") -> JellyfinClient:
    global _jellyfin_client
    if _jellyfin_client:
        _jellyfin_client.close()
    _jellyfin_client = JellyfinClient(url, token, music_library)
    return _jellyfin_client
