"""Smart genre grouping: an LLM maps a library's raw genre tags to a short list of canonical groups.

Libraries tagged from several sources end up with hundreds of genre tags ("Hip-Hop", "Hip Hop", "Rap/Hip-Hop",
"seen live"...). The groups give the filter panel and prompt analysis something manageable to work with.

It is an optional layer: filtering always works on raw tags, and everything works without groups (setting off,
no LLM, a failed call). Groups are rebuilt after a library sync only when the tag list changed, and then only
the new tags are sent to the LLM unless there are too many of them.
"""

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any

from backend import library_cache
from backend.config import get_config
from backend.llm_client import estimate_cost_for_model, get_llm_client

logger = logging.getLogger(__name__)

BATCH_SIZE = 250  # Tags per LLM call, to keep the response within every provider's output limit
REGENERATE_ABOVE = 0.3  # Rebuild all groups when more than this share of the tags is new
OTHER = "Other"

_lock = threading.Lock()

RULES = """Rules:
- Put spelling, casing and naming variants of the same genre in one group ("Alt Rock", "Alternative Rock",
  "Indie/Alternative" -> "Alternative Rock").
- Name groups the way listeners do ("R&B", not "Contemporary R&B").
- A tag joining genres with "/" goes to the group that fits it best.
- A real genre that fits no group goes to a group named "Other".
- Tags that are not genres (moods, descriptions, personal labels such as "melancholy", "seen live",
  "albums I must have") go in "non_genre", not in a group.
- Kinds of recording that people browse like genres are genres: Holiday, Comedy, Spoken Word, Children's,
  Soundtrack, Meditation.
- Use every tag exactly once, copied exactly as given.

Return ONLY valid JSON, no markdown:
{"groups": [{"name": "...", "description": "one short sentence", "genres": ["tag", ...]}], "non_genre": ["tag", ...]}"""

FULL_SYSTEM = f"""You are a music librarian tidying the genre tags of a personal music library.

You get every genre tag in the library with its track count. Sort the tags into canonical genre groups.
Aim for about 30 to 50 groups in a typical library, fewer when there are few tags. Be more specific where the
library is deep (a jazz-heavy library gets "Bebop", "Fusion" and "Smooth Jazz", not one "Jazz" group).

{RULES}"""

ADD_SYSTEM = f"""You are a music librarian tidying the genre tags of a personal music library.

The library already has the genre groups listed below. Sort each new tag into the existing group that fits it.
Only create a new group when none fits, and give it a description.

{RULES}"""


def is_enabled() -> bool:
    return get_config().genre_grouping


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# Storage
# =============================================================================


def _track_tags() -> list[list[str]]:
    """The genre tags of every cached track."""
    conn = library_cache.ensure_db_initialized()
    try:
        return [json.loads(row["genres"] or "[]") for row in conn.execute("SELECT genres FROM tracks")]
    finally:
        conn.close()


def tag_counts() -> dict[str, int]:
    """Track count per raw genre tag in the cache."""
    counts: dict[str, int] = {}
    for tags in _track_tags():
        for tag in tags:
            counts[tag] = counts.get(tag, 0) + 1
    return counts


def load() -> tuple[dict[str, str], dict[str, str | None]]:
    """The stored groups ({name: description}) and mapping ({tag: group name, or None for a non-genre tag})."""
    conn = library_cache.ensure_db_initialized()
    try:
        groups = {r["name"]: r["description"] for r in conn.execute("SELECT name, description FROM genre_groups")}
        mapping = {r["genre"]: r["group_name"] for r in conn.execute("SELECT genre, group_name FROM genre_mappings")}
    finally:
        conn.close()
    return groups, mapping


def _save(groups: dict[str, str], mapping: dict[str, str | None], tags_hash: str) -> None:
    """Replace the stored groups, keeping only groups that still have a tag."""
    used = {group for group in mapping.values() if group}
    conn = library_cache.ensure_db_initialized()
    try:
        conn.execute("DELETE FROM genre_groups")
        conn.execute("DELETE FROM genre_mappings")
        conn.executemany(
            "INSERT INTO genre_groups (name, description) VALUES (?, ?)",
            [(name, description) for name, description in groups.items() if name in used],
        )
        conn.executemany("INSERT INTO genre_mappings (genre, group_name) VALUES (?, ?)", list(mapping.items()))
        conn.execute("UPDATE sync_state SET genre_hash = ? WHERE id = 1", (tags_hash,))
        conn.commit()
    finally:
        conn.close()


def _sync_state() -> tuple[str | None, dict[str, Any] | None]:
    """The hash the groups were built from, and the last run's status."""
    conn = library_cache.ensure_db_initialized()
    try:
        row = conn.execute("SELECT genre_hash, genre_grouping FROM sync_state WHERE id = 1").fetchone()
    finally:
        conn.close()
    return row["genre_hash"], json.loads(row["genre_grouping"]) if row["genre_grouping"] else None


def _record(state: str, message: str, tokens: int = 0, cost: float = 0.0) -> dict[str, Any]:
    status = {"state": state, "message": message, "tokens": tokens, "cost": cost, "at": _now()}
    conn = library_cache.ensure_db_initialized()
    try:
        conn.execute("UPDATE sync_state SET genre_grouping = ? WHERE id = 1", (json.dumps(status),))
        conn.commit()
    finally:
        conn.close()
    log = logger.warning if state == "error" else logger.info
    log("Genre grouping: %s%s", message, f" ({tokens} tokens, ${cost:.4f})" if tokens else "")
    return status


def _letters(tag: str) -> str:
    """A tag's letters and digits only, lowercased, for matching a tag the AI echoed back imperfectly."""
    return "".join(ch for ch in tag.lower() if ch.isalnum())


def _alphabetical(tags: Any) -> list[str]:
    """Case-insensitive order, with case variants ("Rock", "rock") in a fixed order."""
    return sorted(tags, key=lambda tag: (tag.lower(), tag))


def _hash(tags: Any) -> str:
    return hashlib.sha256("\n".join(sorted(tags)).encode()).hexdigest()


# =============================================================================
# What the rest of the app reads
# =============================================================================


def is_active() -> bool:
    """Groups are switched on and exist, so the UI and prompt analysis can use them."""
    if not is_enabled():
        return False
    conn = library_cache.ensure_db_initialized()
    try:
        return conn.execute("SELECT 1 FROM genre_mappings WHERE group_name IS NOT NULL LIMIT 1").fetchone() is not None
    finally:
        conn.close()


def snapshot() -> dict[str, Any]:
    """Groups as the filter panel shows them, with track counts. Tags not grouped yet appear under "Other"."""
    _, mapping = load()
    members: dict[str, set[str]] = {}
    group_counts: dict[str, int] = {}
    non_genre: set[str] = set()
    untagged = 0
    for tags in _track_tags():
        if not tags:
            untagged += 1
        groups_here = set()
        for tag in tags:
            group = mapping[tag] if tag in mapping else OTHER
            if group is None:
                non_genre.add(tag)
            else:
                members.setdefault(group, set()).add(tag)
                groups_here.add(group)
        for group in groups_here:  # A track with two tags in one group counts once for it
            group_counts[group] = group_counts.get(group, 0) + 1

    names = sorted((name for name in members if name != OTHER), key=str.lower) + [OTHER] * (OTHER in members)
    return {
        "groups": [
            {"name": name, "count": group_counts[name], "genres": _alphabetical(members[name])}
            for name in names
        ],
        "non_genre": _alphabetical(non_genre),
        "untagged": untagged,
    }


def expand(group_names: list[Any], groups: list[dict[str, Any]]) -> list[str]:
    """The raw tags behind the named groups of a snapshot (names matched case-insensitively)."""
    wanted = {name.lower() for name in group_names if isinstance(name, str)}
    return [tag for group in groups if group["name"].lower() in wanted for tag in group["genres"]]


def estimate() -> dict[str, Any]:
    """Rough tokens and cost of grouping every tag in the library from scratch."""
    counts = tag_counts()
    batches = -(-len(counts) // BATCH_SIZE)
    # Measured on a 488-tag library with Gemini: output (reasoning included) ran to about 2.8x the tag listing
    listing = sum(len(tag) + 8 for tag in counts) // 4
    input_tokens = int(listing * 1.3) + 1200 * batches  # tags (some sent twice) + instructions and group lists
    output_tokens = int(listing * 2.8)
    config = get_config().llm
    cost = estimate_cost_for_model(config.model_analysis, input_tokens, output_tokens, config=config)
    return {"genres": len(counts), "tokens": input_tokens + output_tokens, "cost": cost}


def status() -> dict[str, Any] | None:
    if _lock.locked():
        return {"state": "running", "message": "Grouping genres...", "tokens": 0, "cost": 0.0, "at": _now()}
    return _sync_state()[1]


# =============================================================================
# Building the groups
# =============================================================================


class _Usage:
    tokens = 0
    cost = 0.0


def _classify(
    tags: list[str], counts: dict[str, int], groups: dict[str, str], llm_client: Any, usage: _Usage
) -> dict[str, str | None]:
    """One LLM call sorting `tags` into groups. New groups are added to `groups`; tags it skipped are left out."""
    listing = "\n".join(f"{tag}\t{counts.get(tag, 0)}" for tag in tags)
    if groups:
        existing = "\n".join(f"- {name}: {description}" for name, description in groups.items())
        system, prompt = ADD_SYSTEM, f"Existing groups:\n{existing}\n\nNew tags (tag, track count):\n{listing}"
    else:
        system, prompt = FULL_SYSTEM, f"Genre tags (tag, track count):\n{listing}"

    # Sorting tags needs little reasoning: on Gemini 3.8 Flash, default thinking took 85 s and ~11k hidden
    # tokens for 250 tags, low effort 16 s and none, with the same groups
    response = llm_client.analyze(prompt, system, effort="low")
    usage.tokens += response.total_tokens
    usage.cost += response.estimated_cost()
    data = llm_client.parse_json_response(response)
    if not isinstance(data, dict):
        raise ValueError("the AI did not return a JSON object")

    # Tags are matched exactly first: "Rock" and "rock" are different tags, and both must be placed. Then
    # loosely, as models sometimes echo a tag back with different case, spacing or escaping ("Blues\\Blues").
    unplaced = set(tags)
    by_lower: dict[str, list[str]] = {}
    by_letters: dict[str, list[str]] = {}
    for tag in tags:
        by_lower.setdefault(tag.lower(), []).append(tag)
        by_letters.setdefault(_letters(tag), []).append(tag)
    known_groups = {name.lower(): name for name in groups}
    result: dict[str, str | None] = {}

    def place(tag: Any, group: str | None) -> bool:
        if not isinstance(tag, str):
            return False
        tag = tag.strip()
        if tag not in unplaced:
            candidates = by_lower.get(tag.lower(), []) + by_letters.get(_letters(tag), [])
            tag = next((t for t in candidates if t in unplaced), None)
        if tag is None:
            return False
        unplaced.discard(tag)
        result[tag] = group
        return True

    for group in data.get("groups") or []:
        if not isinstance(group, dict) or not isinstance(group.get("name"), str) or not group["name"].strip():
            continue
        name = known_groups.setdefault(group["name"].strip().lower(), group["name"].strip())
        if any([place(tag, name) for tag in group.get("genres") or []]):
            groups.setdefault(name, str(group.get("description") or "").strip())
    for tag in data.get("non_genre") or []:
        place(tag, None)
    return result


def _classify_all(
    tags: list[str], counts: dict[str, int], groups: dict[str, str], llm_client: Any, usage: _Usage
) -> dict[str, str | None]:
    """Sort every tag, in batches, most common first so the first batch shapes the groups.

    Tags the LLM skips get one more try, then go to "Other".
    """
    ordered = sorted(tags, key=lambda tag: (-counts.get(tag, 0), tag.lower()))
    mapping: dict[str, str | None] = {}
    for start in range(0, len(ordered), BATCH_SIZE):
        mapping.update(_classify(ordered[start:start + BATCH_SIZE], counts, groups, llm_client, usage))

    missed = [tag for tag in ordered if tag not in mapping]
    if missed:
        try:
            for start in range(0, len(missed), BATCH_SIZE):
                mapping.update(_classify(missed[start:start + BATCH_SIZE], counts, groups, llm_client, usage))
        except Exception as e:
            logger.warning("Genre grouping: retry for %d skipped tags failed: %s", len(missed), e)
        missed = [tag for tag in ordered if tag not in mapping]
    if missed:
        groups.setdefault(OTHER, "Genres that don't fit another group")
        mapping.update(dict.fromkeys(missed, OTHER))
    return mapping


def update(full: bool = False) -> dict[str, Any] | None:
    """Bring the groups up to date with the library's tags. Returns the run's status, or None if it didn't run.

    full=True regroups every tag from scratch (the "Regenerate Genre Groups" button).
    """
    if not is_enabled() or not library_cache.has_cached_tracks():
        return None
    if not _lock.acquire(blocking=False):
        return None
    usage = _Usage()
    try:
        counts = tag_counts()
        tags_hash = _hash(counts)
        stored_hash, _ = _sync_state()
        if tags_hash == stored_hash and not full:
            return _record("skipped", "no changes detected, skipped")

        groups, stored = load()
        mapping = {tag: group for tag, group in stored.items() if tag in counts}
        removed = len(stored) - len(mapping)
        added = [tag for tag in counts if tag not in mapping]
        if not counts:
            _save({}, {}, tags_hash)
            return _record("ok", "the library has no genre tags")
        if not added and not full:
            _save(groups, mapping, tags_hash)
            return _record("ok", f"{removed} removed genre tags dropped, no AI call needed")

        llm_client = get_llm_client()
        if not llm_client:
            return _record("error", "no AI provider is configured")

        many_new = bool(mapping) and len(added) > REGENERATE_ABOVE * len(counts)
        if full or many_new or not mapping:
            groups, mapping = {}, {}
            mapping = _classify_all(list(counts), counts, groups, llm_client, usage)
            if many_new and not full:
                message = "large number of new genres detected, genre groups were fully regenerated"
            else:
                message = "genre groups generated"
        else:
            mapping.update(_classify_all(added, counts, groups, llm_client, usage))
            message = f"{len(added)} new genre tags grouped"

        _save(groups, mapping, tags_hash)
        group_count = len({group for group in mapping.values() if group})
        return _record("ok", f"{message}: {len(counts)} tags in {group_count} groups", usage.tokens, usage.cost)
    except Exception as e:
        # The previous groups stay in place
        return _record("error", f"failed, keeping the previous groups: {e}", usage.tokens, usage.cost)
    finally:
        _lock.release()
