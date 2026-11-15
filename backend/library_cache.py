"""Local SQLite cache for Plex library track metadata.

This module provides fast local access to track data by caching Plex library
metadata in a SQLite database. It eliminates the 2+ minute cold start time
for large libraries by syncing once and loading from cache thereafter.
"""

import json
import logging
import random
import re
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Database location
DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "library_cache.db"

# Patterns for detecting live recordings (same as plex_client.py)
DATE_PATTERN = r"\d{4}[-/]\d{2}[-/]\d{2}"
LIVE_KEYWORDS = r"\b(?:live|concert|sbd|bootleg)\b"

# Batch size for sync operations (smaller = more frequent progress updates)
SYNC_BATCH_SIZE = 500

# Module-level sync state (in-memory for progress tracking)
_sync_state = {
    "is_syncing": False,
    "phase": None,  # "fetching_albums", "fetching", or "processing"
    "current": 0,
    "total": 0,
    "error": None,
    # What the last attempt was and when (in-memory; drives the retry backoff and the status endpoint)
    "kind": None,            # "full" or "incremental"
    "last_attempt_at": None,
    "error_at": None,
}

# Lock to prevent race conditions when starting sync
_sync_lock = threading.Lock()

# Switching media servers: stop a running sync, ignore the old server's cache until replaced
_cancel_sync = threading.Event()
_replace_lock = threading.Lock()
_cache_from_previous_server = False
_switch_count = 0


class SyncCancelled(Exception):
    """Raised inside sync_library when replace_library() asks it to stop."""

# Track if schema has been initialized
_schema_initialized = False
_schema_lock = threading.Lock()


def _is_live_version(title: str, album: str) -> bool:
    """Check if track appears to be a live recording based on title/album."""
    for text in [title, album]:
        if re.search(DATE_PATTERN, text):
            return True
        if re.search(LIVE_KEYWORDS, text, re.IGNORECASE):
            return True
    return False


def get_db_connection() -> sqlite3.Connection:
    """Get a database connection with WAL mode enabled.

    Returns:
        sqlite3.Connection with row_factory set for dict-like access
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row

    # Enable WAL mode for concurrent reads during writes
    conn.execute("PRAGMA journal_mode=WAL")
    # Set busy timeout for lock contention
    conn.execute("PRAGMA busy_timeout=5000")
    # Enable foreign keys (good practice)
    conn.execute("PRAGMA foreign_keys=ON")

    return conn


def init_schema(conn: sqlite3.Connection) -> bool:
    """Initialize database schema if not exists.

    Args:
        conn: Database connection

    Returns:
        True if a schema migration was applied (existing tracks need re-sync),
        False if schema was already up-to-date or freshly created.
    """
    conn.executescript("""
        -- Tracks table: cached Plex track metadata
        CREATE TABLE IF NOT EXISTS tracks (
            rating_key TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            artist TEXT NOT NULL,
            album TEXT NOT NULL,
            duration_ms INTEGER,
            year INTEGER,
            genres TEXT,
            user_rating INTEGER,
            is_live BOOLEAN,
            parent_rating_key TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Indexes for common query patterns
        CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist);
        CREATE INDEX IF NOT EXISTS idx_tracks_year ON tracks(year);
        CREATE INDEX IF NOT EXISTS idx_tracks_is_live ON tracks(is_live);

        -- Sync state: single-row metadata table
        CREATE TABLE IF NOT EXISTS sync_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            plex_server_id TEXT,
            last_sync_at TIMESTAMP,
            track_count INTEGER DEFAULT 0,
            sync_duration_ms INTEGER
        );

        -- Ensure sync_state has exactly one row
        INSERT OR IGNORE INTO sync_state (id) VALUES (1);

        -- Results table: persistent storage for generated playlists and recommendations
        CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            prompt TEXT NOT NULL,
            snapshot JSON NOT NULL,
            track_count INTEGER NOT NULL,
            artist TEXT,
            art_rating_key TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_results_type_created ON results(type, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_results_created_at ON results(created_at DESC);
    """)

    # Migration: add parent_rating_key column if missing (for pre-existing databases)
    migrated = False
    try:
        conn.execute("ALTER TABLE tracks ADD COLUMN parent_rating_key TEXT")
        migrated = True
        logger.info("Migration applied: added parent_rating_key column")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: add view_count and last_viewed_at columns for familiarity tracking
    try:
        conn.execute("ALTER TABLE tracks ADD COLUMN view_count INTEGER DEFAULT 0")
        migrated = True
        logger.info("Migration applied: added view_count column")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        conn.execute("ALTER TABLE tracks ADD COLUMN last_viewed_at TEXT")
        migrated = True
        logger.info("Migration applied: added last_viewed_at column")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: add subtitle column to results table
    try:
        conn.execute("ALTER TABLE results ADD COLUMN subtitle TEXT")
        logger.info("Migration applied: added subtitle column to results")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        conn.execute("ALTER TABLE sync_state ADD COLUMN cache_version INTEGER DEFAULT 1")
        logger.info("Migration applied: added cache_version to sync_state")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # sync_started_at = a full sync is running (still set at startup = the process died mid-sync,
    # re-sync); incremental_at = watermark of the last successful sync (full or incremental) for the next
    # incremental one. Neither is a data migration (no re-sync just for adding them).
    # library_id = the music library the cache was synced from (NULL on caches older than the column).
    for column in ("sync_started_at TEXT", "incremental_at TEXT", "library_id TEXT"):
        try:
            conn.execute(f"ALTER TABLE sync_state ADD COLUMN {column}")
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Index on parent_rating_key (must come after migration adds the column)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tracks_parent_key ON tracks(parent_rating_key)")

    conn.commit()
    return migrated


# Whether a migration was applied on startup (signals need for re-sync)
_migration_applied = False

# Bump when sync starts storing different data, so existing caches re-sync on upgrade.
# 2: Plex genres combine album, artist and track genres (album only before)
CACHE_VERSION = 2


def ensure_db_initialized() -> sqlite3.Connection:
    """Ensure database exists and schema is initialized.

    Returns:
        Initialized database connection
    """
    global _schema_initialized, _migration_applied
    conn = get_db_connection()

    # Only initialize schema once per process; lock prevents races on startup
    if not _schema_initialized:
        with _schema_lock:
            if not _schema_initialized:
                migrated = init_schema(conn)
                # A new or empty database has nothing to rebuild
                has_tracks = conn.execute("SELECT 1 FROM tracks LIMIT 1").fetchone() is not None
                _migration_applied = migrated and has_tracks
                _schema_initialized = True

    return conn


def get_sync_state() -> dict[str, Any]:
    """Get current sync state from database and in-memory state.

    Returns:
        Dict with track_count, synced_at, is_syncing, sync_progress, error
    """
    conn = ensure_db_initialized()
    try:
        row = conn.execute(
            "SELECT plex_server_id, library_id, last_sync_at, track_count, sync_duration_ms, sync_started_at, "
            "incremental_at FROM sync_state WHERE id = 1"
        ).fetchone()

        # Snapshot sync state under lock for consistent reads
        with _sync_lock:
            ss = dict(_sync_state)

        result = {
            "track_count": row["track_count"] if row else 0,
            "synced_at": row["last_sync_at"] if row else None,
            "plex_server_id": row["plex_server_id"] if row else None,
            "library_id": row["library_id"] if row else None,
            "sync_duration_ms": row["sync_duration_ms"] if row else None,
            "is_syncing": ss["is_syncing"],
            "sync_progress": None,
            "error": ss["error"],
            "sync_kind": ss["kind"],
            "last_attempt_at": ss["last_attempt_at"],
            "error_at": ss["error_at"],
            "incremental_at": row["incremental_at"] if row else None,
            "interrupted": bool(row and row["sync_started_at"] and not ss["is_syncing"]),
        }

        if ss["is_syncing"]:
            result["sync_progress"] = {
                "phase": ss["phase"],
                "current": ss["current"],
                "total": ss["total"],
            }

        return result
    finally:
        conn.close()


def get_cached_tracks() -> list[dict[str, Any]]:
    """Get all tracks from cache.

    Returns:
        List of track dicts with all fields
    """
    conn = ensure_db_initialized()
    try:
        rows = conn.execute(
            "SELECT rating_key, title, artist, album, duration_ms, year, "
            "genres, user_rating, is_live FROM tracks"
        ).fetchall()

        tracks = []
        for row in rows:
            track = dict(row)
            # Parse genres JSON
            if track["genres"]:
                track["genres"] = json.loads(track["genres"])
            else:
                track["genres"] = []
            tracks.append(track)

        return tracks
    finally:
        conn.close()


def _artist_conditions(
    artists: list[str] | None, exclude_artists: list[str] | None, params: list[Any]
) -> list[str]:
    """SQL conditions for the artist filters (case-insensitive), adding their params."""
    conditions = []
    for names, operator in ((artists, "IN"), (exclude_artists, "NOT IN")):
        if names:
            conditions.append(f"LOWER(artist) {operator} ({', '.join('?' for _ in names)})")
            params.extend(name.lower() for name in names)
    return conditions


def search_artists(query: str = "", limit: int = 20) -> list[dict[str, Any]]:
    """Artists in the cache whose name contains query, with track counts, most tracks first."""
    conn = ensure_db_initialized()
    try:
        rows = conn.execute(
            "SELECT artist AS name, COUNT(*) AS count FROM tracks "
            "WHERE artist LIKE ? GROUP BY artist ORDER BY count DESC, artist LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_artist_names() -> list[str]:
    """Every distinct artist name in the cache."""
    conn = ensure_db_initialized()
    try:
        return [row[0] for row in conn.execute("SELECT DISTINCT artist FROM tracks")]
    finally:
        conn.close()


def get_tracks_by_filters(
    genres: list[str] | None = None,
    decades: list[str] | None = None,
    min_rating: int = 0,
    exclude_live: bool = True,
    limit: int = 0,
    artists: list[str] | None = None,
    exclude_artists: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Get tracks from cache matching filter criteria.

    Args:
        genres: List of genre names to include (OR matching)
        decades: List of decades like "1990s" (OR matching)
        min_rating: Minimum user rating (0-10, 0 = no filter)
        exclude_live: Whether to exclude live recordings
        limit: Max tracks to return (0 = no limit)
        artists: Only tracks by these artists
        exclude_artists: No tracks by these artists

    Returns:
        List of matching track dicts
    """
    conn = ensure_db_initialized()
    try:
        conditions = []
        params: list[Any] = []

        if exclude_live:
            conditions.append("is_live = 0")

        if min_rating > 0:
            conditions.append("user_rating >= ?")
            params.append(min_rating)

        if decades:
            decade_conditions = []
            for decade in decades:
                # Convert "1990s" to year range
                try:
                    start_year = int(decade.rstrip("s"))
                except ValueError:
                    continue
                end_year = start_year + 9
                decade_conditions.append("(year >= ? AND year <= ?)")
                params.extend([start_year, end_year])
            if decade_conditions:
                conditions.append(f"({' OR '.join(decade_conditions)})")

        conditions.extend(_artist_conditions(artists, exclude_artists, params))

        # Build query
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM tracks WHERE {where_clause}"

        # Only apply SQL LIMIT when no genre filter (genre filtering happens in Python)
        # If genres are specified, we need all matching tracks first, then filter, then sample
        if limit > 0 and not genres:
            query += " ORDER BY RANDOM() LIMIT ?"
            params.append(limit)

        rows = conn.execute(query, params).fetchall()
        tracks = []

        for row in rows:
            track = dict(row)
            # Parse genres JSON
            if track["genres"]:
                track["genres"] = json.loads(track["genres"])
            else:
                track["genres"] = []

            # Genre filtering in Python (JSON field doesn't support SQL IN)
            if genres:
                track_genres = [g.lower() for g in track["genres"]]
                if not any(g.lower() in track_genres for g in genres):
                    continue

            tracks.append(track)

        # Apply limit after genre filtering with random sampling
        if limit > 0 and genres and len(tracks) > limit:
            tracks = random.sample(tracks, limit)

        return tracks
    finally:
        conn.close()


def clear_cache() -> None:
    """Clear all cached tracks and reset sync state."""
    conn = ensure_db_initialized()
    try:
        conn.execute("DELETE FROM tracks")
        conn.execute(
            "UPDATE sync_state SET last_sync_at = NULL, track_count = 0, "
            "sync_duration_ms = NULL WHERE id = 1"
        )
        conn.commit()
        logger.info("Cache cleared")
    finally:
        conn.close()


def is_cache_stale(max_age_hours: float = 24) -> bool:
    """Check if cache is older than max_age_hours.

    Args:
        max_age_hours: Maximum cache age in hours

    Returns:
        True if cache is stale or empty
    """
    state = get_sync_state()
    if not state["synced_at"] or not state["track_count"]:
        return True   # an empty cache is stale whatever its timestamp says

    try:
        # Parse ISO timestamp
        synced_at = datetime.fromisoformat(state["synced_at"].replace("Z", "+00:00"))
        age_hours = (datetime.now(timezone.utc) - synced_at).total_seconds() / 3600
        return age_hours > max_age_hours
    except (ValueError, TypeError):
        return True


def check_server_changed(current_server_id: str, current_library_id: str | None = None) -> bool:
    """Check if the cache holds a different server's library, or a different music library on the same server.

    Args:
        current_server_id: Current server's identifier (Plex machineIdentifier, Jellyfin server Id)
        current_library_id: Current music library's identifier, if the client knows it

    Returns:
        True if the server or library changed (cache should be cleared)
    """
    state = get_sync_state()
    cached_server_id = state.get("plex_server_id")

    if not cached_server_id:
        return False  # First sync, no change
    if cached_server_id != current_server_id:
        return True
    # An unknown library on either side is not a change (caches synced before library_id was stored)
    cached_library_id = state.get("library_id")
    return bool(cached_library_id and current_library_id and cached_library_id != current_library_id)


def library_id_of(media_client: Any) -> str | None:
    """The ID of the music library a client is set to, or None if it has none."""
    get_id = getattr(media_client, "get_library_identifier", None)
    return get_id() if get_id else None


_INSERT_TRACK_SQL = (
    "INSERT OR REPLACE INTO tracks "
    "(rating_key, title, artist, album, duration_ms, year, genres, "
    "user_rating, is_live, parent_rating_key, view_count, last_viewed_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def _merge_genres(*genre_lists: list[str]) -> list[str]:
    """Combine genre lists in order, dropping case-insensitive duplicates."""
    merged: dict[str, str] = {}
    for genres in genre_lists:
        for genre in genres:
            merged.setdefault(genre.lower(), genre)
    return list(merged.values())


def _track_genre_tags(raw: dict[str, Any]) -> list[str]:
    """A track's own genre tags, from the listing row's XML.

    plexapi's track.genres is a lazy property: it is not in vars(track), and reading it on a row without
    tags costs a request.
    """
    data = raw.get("_data")
    if data is not None:
        return [el.attrib["tag"] for el in data if el.tag == "Genre" and el.attrib.get("tag")]
    return [g.tag for g in raw.get("genres") or []]


def _plex_track_row(
    album_metadata: dict[str, dict[str, Any]], artist_genres: dict[str, list[str]], track: Any
) -> tuple:
    """Build a tracks-table row from a raw plexapi track.

    The year comes from its album. Genres combine the album's, the artist's (where people
    usually curate their own) and the track's own tags.

    Fields are read from vars(track), never as attributes - on a partial plexapi object (anything
    from a section listing) an attribute that is None or [] triggers one GET /library/metadata/<key> per track.
    """
    raw = vars(track)
    title = raw.get("title") or ""
    album = raw.get("parentTitle") or ""
    parent_key = str(raw.get("parentRatingKey") or "")
    album_data = album_metadata.get(parent_key, {})
    genres = _merge_genres(
        album_data.get("genres", []),
        artist_genres.get(str(raw.get("grandparentRatingKey") or ""), []),
        _track_genre_tags(raw),
    )
    last_viewed_at = raw.get("lastViewedAt")
    return (
        str(raw.get("ratingKey")),
        title,
        raw.get("grandparentTitle") or "Unknown Artist",
        album,
        raw.get("duration") or 0,
        album_data.get("year"),
        json.dumps(genres),
        raw.get("userRating"),
        _is_live_version(title, album),
        parent_key,
        raw.get("viewCount") or 0,
        last_viewed_at.isoformat() if last_viewed_at else None,
    )


def _jellyfin_track_row(row: dict[str, Any]) -> tuple:
    """Build a tracks-table row from a JellyfinClient.get_all_tracks_for_sync() entry."""
    track = row["track"]
    return (
        track.rating_key,
        track.title,
        track.artist or "Unknown Artist",
        track.album,
        track.duration_ms,
        track.year,
        json.dumps(track.genres),
        track.user_rating,
        _is_live_version(track.title, track.album),
        row["album_key"],
        row["view_count"],
        row["last_viewed_at"],
    )


INCREMENTAL_MARGIN_S = 15 * 60   # re-read tracks changed shortly before the watermark (upserts are idempotent)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def incremental_due(max_age_minutes: float) -> bool:
    """True when the cache has tracks and the last (full or incremental) sync is older than max_age_minutes."""
    state = get_sync_state()
    if not state["track_count"]:
        return False   # nothing to add to; needs a full sync first
    last = _parse_ts(state["incremental_at"]) or _parse_ts(state["synced_at"])
    if last is None:
        return True
    return (datetime.now(timezone.utc) - last).total_seconds() > max_age_minutes * 60


class _PlexMetadata:
    """Album year/genres + artist genres for the tracks seen so far, fetched in batches as pages arrive."""

    def __init__(self, media_client: Any):
        self._client = media_client
        self.albums: dict[str, dict[str, Any]] = {}
        self.artists: dict[str, list[str]] = {}

    def load_for(self, page: list[Any]) -> None:
        album_keys, artist_keys = set(), set()
        for track in page:
            raw = vars(track)
            album_keys.add(str(raw.get("parentRatingKey") or ""))
            artist_keys.add(str(raw.get("grandparentRatingKey") or ""))
        album_keys -= set(self.albums)
        artist_keys -= set(self.artists)
        if album_keys:
            found = self._client.fetch_metadata(sorted(album_keys))
            for key in album_keys:   # missing from Plex (deleted meanwhile): remember the miss, no refetch
                self.albums[key] = found.get(key, {})
        if artist_keys:
            found = self._client.fetch_metadata(sorted(artist_keys))
            for key in artist_keys:
                self.artists[key] = found.get(key, {}).get("genres", [])


def sync_library(
    media_client: Any,
    on_progress: Callable[[int, int], None] | None = None,
    incremental: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Sync tracks from Plex or Jellyfin to local cache.

    This is a blocking synchronous operation. For async usage, wrap in
    asyncio.to_thread() or run in a thread pool.

    Notes:
    - Plex tracks are read page by page (PlexClient.iter_raw_tracks) with batched album/artist metadata, so a sync
      holds one page in memory instead of the whole library and makes ~1 request per 5,000 tracks instead of one
      per track (hours -> minutes).
    - incremental=True (Plex only) upserts tracks added or changed since the last sync's watermark instead of
      rebuilding the cache; deletions are picked up by the next full sync. Jellyfin keeps the upstream path.
    - A full sync persists sync_started_at before it starts and clears it when it commits or fails; a process
      killed mid-sync leaves it set and sync_needed() restarts the sync on boot.
    - A full sync that reads far fewer tracks than Plex reports fails and keeps the old cache. So does one that
      would shrink the cache to under half, unless force=True.

    Args:
        media_client: Connected PlexClient or JellyfinClient
        on_progress: Optional callback(current, total) for progress updates
        incremental: only tracks changed since the last sync (needs a previous full sync)
        force: accept a library less than half the size of the cache

    Returns:
        Dict with success, track_count, duration_ms, error
    """
    global _sync_state

    is_plex = hasattr(media_client, "iter_raw_tracks")
    if incremental and not is_plex:
        return {"success": False, "error": "Incremental sync is only available for Plex"}

    # Use lock to prevent race condition between check and set
    with _sync_lock:
        if _sync_state["is_syncing"]:
            return {"success": False, "error": "Sync already in progress"}

        _sync_state = {
            "is_syncing": True,
            "phase": "fetching_albums",
            "current": 0,
            "total": 0,
            "error": None,
            "kind": "incremental" if incremental else "full",
            "last_attempt_at": _now_iso(),
            "error_at": None,
        }

    start_time = time.time()
    started_at = datetime.now(timezone.utc)
    conn = None
    label = "Incremental sync" if incremental else "Sync"

    try:
        # Get server ID for cache validation
        server_id = media_client.get_machine_identifier()
        if not server_id:
            raise ValueError("Could not get media server identifier")

        # Check if the server or music library changed - clear cache if so
        library_id = library_id_of(media_client)
        if check_server_changed(server_id, library_id):
            logger.info("Media server or music library changed, clearing cache")
            clear_cache()

        conn = ensure_db_initialized()
        previous_state = get_sync_state()
        previous = previous_state["track_count"] or 0

        if incremental:
            if not previous:
                raise RuntimeError("no full sync yet - nothing to update incrementally")
            watermark = _parse_ts(previous_state["incremental_at"]) or _parse_ts(previous_state["synced_at"])
            since = int(watermark.timestamp()) - INCREMENTAL_MARGIN_S if watermark else None
            if since is None:
                raise RuntimeError("no sync watermark - run a full sync")
            logger.info("Incremental sync: tracks changed since %s", watermark.isoformat())
        else:
            since = None
            # Mark the full sync as running (own short transaction) - still set at startup = killed mid-sync.
            conn.execute("UPDATE sync_state SET sync_started_at = ? WHERE id = 1", (started_at.isoformat(),))
            conn.commit()

        # Replace the cache in ONE transaction: the delete, the inserts and the
        # sync_state update are committed together at the end. If the sync fails
        # or the process is killed partway (e.g. a container restart), SQLite
        # rolls it back and the previous complete cache is kept. In WAL mode,
        # readers keep seeing that previous snapshot until the commit.
        if not incremental:
            conn.execute("DELETE FROM tracks")

        synced_count = 0
        if is_plex:
            expected = media_client.get_library_total_tracks() if not incremental else 0
            with _sync_lock:
                _sync_state["phase"] = "fetching"
                _sync_state["total"] = expected
            metadata = _PlexMetadata(media_client)
            logger.info("%s: reading tracks from Plex in pages of %d%s", label, media_client.SYNC_PAGE_SIZE,
                        f" ({expected} expected)" if expected else "")
            for page in media_client.iter_raw_tracks(updated_since=since):
                if _cancel_sync.is_set():
                    raise SyncCancelled
                with _sync_lock:
                    _sync_state["phase"] = "processing"
                metadata.load_for(page)
                to_row = partial(_plex_track_row, metadata.albums, metadata.artists)
                conn.executemany(_INSERT_TRACK_SQL, [to_row(track) for track in page])
                synced_count += len(page)
                with _sync_lock:
                    _sync_state["current"] = synced_count
                    if synced_count > _sync_state["total"]:
                        _sync_state["total"] = synced_count
                if on_progress:
                    on_progress(synced_count, expected or synced_count)
                logger.info("%s: %d/%s tracks", label, synced_count, expected or "?")
                del page
            total = synced_count
        else:
            with _sync_lock:
                _sync_state["phase"] = "fetching"
            logger.info("Fetching all tracks from the media server (this may take 30-60s)...")
            all_tracks = media_client.get_all_tracks_for_sync()
            total = len(all_tracks)
            logger.info("Got %d tracks from the media server", total)
            if _cancel_sync.is_set():
                raise SyncCancelled
            with _sync_lock:
                _sync_state["total"] = total
                _sync_state["phase"] = "processing"
            batch_data = []
            for track in all_tracks:
                if _cancel_sync.is_set():
                    raise SyncCancelled
                batch_data.append(_jellyfin_track_row(track))
                if len(batch_data) >= SYNC_BATCH_SIZE:
                    conn.executemany(_INSERT_TRACK_SQL, batch_data)
                    synced_count += len(batch_data)
                    batch_data = []
                    with _sync_lock:
                        _sync_state["current"] = synced_count
                    if on_progress:
                        on_progress(synced_count, total)
                    logger.info("Synced %d/%d tracks", synced_count, total)
            if batch_data:
                conn.executemany(_INSERT_TRACK_SQL, batch_data)
                synced_count += len(batch_data)
                with _sync_lock:
                    _sync_state["current"] = synced_count

        # A short read or a cache shrinking to under half is a failure, so the rollback keeps the cache
        # (force accepts the smaller library). A library that really is empty syncs to 0 tracks.
        if not incremental:
            if is_plex and expected and total < expected * 0.5:
                raise RuntimeError(f"read {total} of {expected} tracks Plex reports - keeping the cache")
            if not force and previous and total < previous * 0.5:
                raise RuntimeError(
                    f"media server returned {total} tracks (cache has {previous}) - keeping the cache. "
                    "Refresh the library to accept it"
                )

        duration_ms = int((time.time() - start_time) * 1000)
        synced_at = _now_iso()
        if incremental:
            track_count = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
            conn.execute(
                "UPDATE sync_state SET incremental_at = ?, track_count = ? WHERE id = 1",
                (started_at.isoformat(), track_count),
            )
        else:
            track_count = synced_count
            conn.execute(
                "UPDATE sync_state SET plex_server_id = ?, library_id = ?, last_sync_at = ?, track_count = ?, "
                "sync_duration_ms = ?, cache_version = ?, sync_started_at = NULL, incremental_at = ? WHERE id = 1",
                (server_id, library_id, synced_at, synced_count, duration_ms, CACHE_VERSION,
                 started_at.isoformat()),
            )
        conn.commit()

        logger.info("%s complete: %d tracks in %dms (cache %d)", label, synced_count, duration_ms, track_count)

        # Clear migration flag — new columns are now populated
        global _migration_applied
        if not incremental:
            _migration_applied = False

        return {
            "success": True,
            "track_count": track_count,
            "duration_ms": duration_ms,
        }

    except SyncCancelled:
        logger.info("Sync cancelled: the media server changed")
        if conn:
            conn.rollback()
            _clear_sync_marker(conn)
        return {"success": False, "error": "Sync cancelled"}

    except Exception as e:
        logger.exception("%s failed: %s", label, e)
        if conn:
            conn.rollback()  # keep the previous complete cache
            _clear_sync_marker(conn)
        with _sync_lock:
            _sync_state["error"] = str(e)
            _sync_state["error_at"] = _now_iso()
        return {"success": False, "error": str(e)}

    finally:
        with _sync_lock:
            _sync_state["is_syncing"] = False
            _sync_state["phase"] = None
            _sync_state["current"] = 0
            _sync_state["total"] = 0
        if conn:
            conn.close()


def _clear_sync_marker(conn: sqlite3.Connection) -> None:
    """A full sync that failed (and rolled back) is not an interrupted one: the retry loop handles it."""
    try:
        conn.execute("UPDATE sync_state SET sync_started_at = NULL WHERE id = 1")
        conn.commit()
    except sqlite3.Error:
        logger.exception("Could not clear sync_started_at")


def get_sync_progress() -> dict[str, Any]:
    """Get current sync progress (for polling).

    Returns:
        Dict with is_syncing, phase, current, total, error
    """
    with _sync_lock:
        return dict(_sync_state)


def count_tracks_by_filters(
    genres: list[str] | None = None,
    decades: list[str] | None = None,
    min_rating: int = 0,
    exclude_live: bool = True,
    artists: list[str] | None = None,
    exclude_artists: list[str] | None = None,
) -> int:
    """Count tracks matching filter criteria without fetching full data.

    Args:
        genres: List of genre names to include (OR matching)
        decades: List of decades like "1990s" (OR matching)
        min_rating: Minimum user rating (0-10, 0 = no filter)
        exclude_live: Whether to exclude live recordings
        artists: Only tracks by these artists
        exclude_artists: No tracks by these artists

    Returns:
        Count of matching tracks, or -1 if cache is empty
    """
    state = get_sync_state()
    if state["track_count"] == 0:
        return -1  # Cache empty, signal to use Plex

    conn = ensure_db_initialized()
    try:
        conditions = []
        params: list[Any] = []

        if exclude_live:
            conditions.append("is_live = 0")

        if min_rating > 0:
            conditions.append("user_rating >= ?")
            params.append(min_rating)

        if decades:
            decade_conditions = []
            for decade in decades:
                try:
                    start_year = int(decade.rstrip("s"))
                except ValueError:
                    continue
                end_year = start_year + 9
                decade_conditions.append("(year >= ? AND year <= ?)")
                params.extend([start_year, end_year])
            if decade_conditions:
                conditions.append(f"({' OR '.join(decade_conditions)})")

        conditions.extend(_artist_conditions(artists, exclude_artists, params))

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # No genre filter - use simple count query
        if not genres:
            query = f"SELECT COUNT(*) FROM tracks WHERE {where_clause}"
            count = conn.execute(query, params).fetchone()[0]
            return count

        # Genre filter - need to check JSON field, so fetch and filter in Python
        query = f"SELECT genres FROM tracks WHERE {where_clause}"
        rows = conn.execute(query, params).fetchall()

        count = 0
        genres_lower = [g.lower() for g in genres]
        for row in rows:
            if row["genres"]:
                track_genres = json.loads(row["genres"])
                track_genres_lower = [g.lower() for g in track_genres]
                if any(g in track_genres_lower for g in genres_lower):
                    count += 1

        return count
    finally:
        conn.close()


def has_cached_tracks() -> bool:
    """Check if the cache has tracks from the current media server."""
    if _cache_from_previous_server:
        return False
    return get_sync_state()["track_count"] > 0


def invalidate_cache() -> int:
    """Stop using the cache right away and cancel any sync in progress.

    Called when the media server changes. Returns the switch number to pass to
    replace_library(), which then rebuilds the cache.
    """
    global _cache_from_previous_server, _switch_count
    _cache_from_previous_server = True
    _switch_count += 1
    _cancel_sync.set()
    return _switch_count


def replace_library(get_media_client: Callable[[], Any], switch: int) -> None:
    """Wait out any running sync, clear the cache and sync the current media server.

    After several quick switches only the latest one syncs; earlier ones return early.
    """
    global _cache_from_previous_server
    with _replace_lock:
        if switch != _switch_count:
            return
        while get_sync_progress()["is_syncing"]:
            time.sleep(0.5)
        _cancel_sync.clear()
        clear_cache()
        _cache_from_previous_server = False

        media_client = get_media_client()
        if media_client and media_client.is_connected():
            sync_library(media_client)


def needs_resync() -> bool:
    """Check if the cache must be re-synced: a schema migration, or an older CACHE_VERSION.

    The frontend shows its blocking upgrade dialog on this. An empty or interrupted cache is sync_needed().

    Safe for fresh DBs: _migration_applied is False when CREATE TABLE already includes
    all columns, and an empty cache has nothing to rebuild.
    """
    if _migration_applied:
        return True
    conn = ensure_db_initialized()
    try:
        row = conn.execute("SELECT track_count, cache_version FROM sync_state WHERE id = 1").fetchone()
    finally:
        conn.close()
    return bool(row and row["track_count"] and (row["cache_version"] or 1) < CACHE_VERSION)


def sync_needed() -> str | None:
    """Why the cache needs a full sync whatever its age: "interrupted", "empty", or None.

    Interrupted means the previous process died mid-sync (e.g. a container restart).
    """
    conn = ensure_db_initialized()
    try:
        row = conn.execute("SELECT track_count, sync_started_at FROM sync_state WHERE id = 1").fetchone()
    finally:
        conn.close()
    if not row or get_sync_progress()["is_syncing"]:
        return None
    if row["sync_started_at"]:
        return "interrupted"
    return None if row["track_count"] else "empty"


def get_album_candidates(
    genres: list[str] | None = None,
    decades: list[str] | None = None,
    exclude_live: bool = True,
) -> list[dict[str, Any]]:
    """Get album candidates aggregated from cached tracks.

    Groups tracks by parent_rating_key to produce album-level data.
    Supports genre/decade filtering.

    Args:
        genres: Optional genre filter (OR matching)
        decades: Optional decade filter (OR matching, e.g. "1990s")
        exclude_live: Exclude live recordings (default True)

    Returns:
        List of album dicts with parent_rating_key, album, album_artist,
        year, genres, decade, track_count, track_rating_keys.
    """
    conn = ensure_db_initialized()
    try:
        conditions = ["parent_rating_key IS NOT NULL", "parent_rating_key != ''"]
        if exclude_live:
            conditions.append("is_live = 0")
        params: list[Any] = []

        if decades:
            decade_conditions = []
            for decade in decades:
                try:
                    start_year = int(decade.rstrip("s"))
                except ValueError:
                    continue
                end_year = start_year + 9
                decade_conditions.append("(year >= ? AND year <= ?)")
                params.extend([start_year, end_year])
            if decade_conditions:
                conditions.append(f"({' OR '.join(decade_conditions)})")

        where_clause = " AND ".join(conditions)
        query = (
            f"SELECT rating_key, title, artist, album, year, genres, parent_rating_key "
            f"FROM tracks WHERE {where_clause} "
            f"ORDER BY parent_rating_key, rating_key"
        )

        rows = conn.execute(query, params).fetchall()

        # Aggregate tracks into albums
        albums: dict[str, dict[str, Any]] = {}
        for row in rows:
            prk = row["parent_rating_key"]
            track_genres = json.loads(row["genres"]) if row["genres"] else []

            if prk not in albums:
                # Derive decade from year
                year = row["year"]
                decade = ""
                if year:
                    decade_start = (year // 10) * 10
                    decade = f"{decade_start}s"

                albums[prk] = {
                    "parent_rating_key": prk,
                    "album": row["album"],
                    # artist column stores grandparentTitle (album artist), not track artist
                    "album_artist": row["artist"],
                    "year": year,
                    "genres": [],
                    "decade": decade,
                    "track_count": 0,
                    "track_rating_keys": [],
                    "_genre_set": set(),
                }

            album = albums[prk]
            album["track_count"] += 1
            album["track_rating_keys"].append(row["rating_key"])
            for g in track_genres:
                if g not in album["_genre_set"]:
                    album["_genre_set"].add(g)
                    album["genres"].append(g)

        # Apply genre filter in Python (genres stored as JSON)
        result = []
        genres_lower = [g.lower() for g in genres] if genres else None
        for album in albums.values():
            # Remove internal tracking set
            del album["_genre_set"]

            if genres_lower:
                album_genres_lower = [g.lower() for g in album["genres"]]
                if not any(g in album_genres_lower for g in genres_lower):
                    continue

            result.append(album)

        return result
    finally:
        conn.close()


def get_cached_genre_decade_stats() -> dict[str, list[dict[str, Any]]]:
    """Get genre and decade stats from the local cache.

    Returns genre/decade lists derived from cached tracks, avoiding a
    round-trip to the Plex server.

    Returns:
        Dict with 'genres' and 'decades' lists, each containing
        {'name': str, 'count': int} dicts sorted by name.
    """
    conn = ensure_db_initialized()
    try:
        rows = conn.execute("SELECT genres, year FROM tracks").fetchall()

        genre_counts: dict[str, int] = {}
        decade_counts: dict[str, int] = {}

        for row in rows:
            # Tally genres
            if row["genres"]:
                for g in json.loads(row["genres"]):
                    genre_counts[g] = genre_counts.get(g, 0) + 1

            # Tally decades
            year = row["year"]
            if year:
                decade_start = (year // 10) * 10
                decade_name = f"{decade_start}s"
                decade_counts[decade_name] = decade_counts.get(decade_name, 0) + 1

        genres = sorted(
            [{"name": name, "count": count} for name, count in genre_counts.items()],
            key=lambda x: x["name"],
        )
        decades = sorted(
            [{"name": name, "count": count} for name, count in decade_counts.items()],
            key=lambda x: x["name"],
        )

        return {"genres": genres, "decades": decades}
    finally:
        conn.close()


def get_album_familiarity(
    parent_rating_keys: list[str] | None = None,
) -> dict[str, dict]:
    """Get familiarity data for albums aggregated from cached track play history.

    Classifies each album as:
    - "unplayed": 0 total plays across all tracks
    - "well-loved": avg plays per track >= 3
    - "light": some plays but avg < 3

    Args:
        parent_rating_keys: Optional list of album keys to query.
            If None, returns all albums.

    Returns:
        Dict mapping parent_rating_key -> {"level": str, "last_viewed_at": str|None}
    """
    conn = ensure_db_initialized()
    try:
        query = (
            "SELECT parent_rating_key, "
            "SUM(view_count) AS total_plays, "
            "AVG(view_count) AS avg_plays, "
            "MAX(last_viewed_at) AS last_viewed "
            "FROM tracks "
            "WHERE parent_rating_key IS NOT NULL AND parent_rating_key != '' "
        )
        params: list[str] = []

        if parent_rating_keys is not None:
            placeholders = ",".join("?" for _ in parent_rating_keys)
            query += f"AND parent_rating_key IN ({placeholders}) "
            params.extend(parent_rating_keys)

        query += "GROUP BY parent_rating_key"

        rows = conn.execute(query, params).fetchall()

        result: dict[str, dict] = {}
        for row in rows:
            total_plays = row["total_plays"] or 0
            avg_plays = row["avg_plays"] or 0

            if total_plays == 0:
                level = "unplayed"
            elif avg_plays >= 3:
                level = "well-loved"
            else:
                level = "light"

            result[row["parent_rating_key"]] = {
                "level": level,
                "last_viewed_at": row["last_viewed"],
            }

        return result
    finally:
        conn.close()


# =============================================================================
# Results Persistence
# =============================================================================


def save_result(
    result_type: str,
    title: str,
    prompt: str,
    snapshot: dict,
    track_count: int,
    artist: str | None = None,
    art_rating_key: str | None = None,
    subtitle: str | None = None,
) -> str:
    """Save a generated result and return its unique ID.

    Args:
        result_type: "prompt_playlist", "seed_playlist", or "album_recommendation"
        title: Display title for the result
        prompt: Original user prompt
        snapshot: Full serialized response (GenerateResponse or RecommendGenerateResponse)
        track_count: Number of tracks in the result
        artist: Primary artist (for album recs)
        art_rating_key: Rating key for thumbnail art
        subtitle: Pre-computed subtitle for history feed cards

    Returns:
        16-char hex ID for the saved result
    """
    conn = ensure_db_initialized()
    try:
        # Generate collision-resistant ID with INSERT OR IGNORE to handle
        # concurrent inserts that race past the existence check.
        for _ in range(10):
            result_id = secrets.token_hex(8)
            cursor = conn.execute(
                """INSERT OR IGNORE INTO results (id, type, title, prompt, snapshot, track_count, artist, art_rating_key, subtitle)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (result_id, result_type, title, prompt, json.dumps(snapshot), track_count, artist, art_rating_key, subtitle),
            )
            if cursor.rowcount > 0:
                break
        else:
            raise RuntimeError("Failed to generate unique result ID after 10 attempts")
        conn.commit()
        logger.info("Saved result %s (type=%s, tracks=%d)", result_id, result_type, track_count)
        return result_id
    finally:
        conn.close()


def get_result(result_id: str) -> dict[str, Any] | None:
    """Fetch a single result by ID, including its snapshot.

    Returns:
        Dict with all columns (snapshot parsed from JSON), or None if not found.
    """
    conn = ensure_db_initialized()
    try:
        row = conn.execute(
            "SELECT id, type, title, prompt, snapshot, track_count, artist, art_rating_key, subtitle, created_at FROM results WHERE id = ?",
            (result_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "type": row["type"],
            "title": row["title"],
            "prompt": row["prompt"],
            "snapshot": json.loads(row["snapshot"]),
            "track_count": row["track_count"],
            "artist": row["artist"],
            "art_rating_key": row["art_rating_key"],
            "subtitle": row["subtitle"],
            "created_at": row["created_at"],
        }
    finally:
        conn.close()


def list_results(
    result_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """List results ordered by created_at DESC, without snapshots.

    Args:
        result_type: Optional type filter (can be comma-separated for multiple)
        limit: Max results to return
        offset: Pagination offset

    Returns:
        Tuple of (list of result dicts without snapshot, total count)
    """
    conn = ensure_db_initialized()
    try:
        where_clause = ""
        params: list[Any] = []

        if result_type:
            types = [t.strip() for t in result_type.split(",") if t.strip()]
            placeholders = ",".join("?" for _ in types)
            where_clause = f"WHERE type IN ({placeholders})"
            params = types

        # Get total count
        total = conn.execute(
            f"SELECT COUNT(*) FROM results {where_clause}", params
        ).fetchone()[0]

        # Get page
        rows = conn.execute(
            f"""SELECT id, type, title, prompt, track_count, artist, art_rating_key, subtitle, created_at
                FROM results {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?""",
            params + [limit, offset],
        ).fetchall()

        results = [
            {
                "id": row["id"],
                "type": row["type"],
                "title": row["title"],
                "prompt": row["prompt"],
                "track_count": row["track_count"],
                "artist": row["artist"],
                "art_rating_key": row["art_rating_key"],
                "subtitle": row["subtitle"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        return results, total
    finally:
        conn.close()


def delete_result(result_id: str) -> bool:
    """Delete a result by ID.

    Returns:
        True if a row was deleted, False if not found.
    """
    conn = ensure_db_initialized()
    try:
        cursor = conn.execute("DELETE FROM results WHERE id = ?", (result_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted result %s", result_id)
        return deleted
    finally:
        conn.close()
