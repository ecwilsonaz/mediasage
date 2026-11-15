"""Prompt analysis and seed track dimension extraction."""

from rapidfuzz import fuzz, process
from unidecode import unidecode

from backend import library_cache
from backend.config import get_current_media_client
from backend.llm_client import get_llm_client
from backend.models import (
    AnalyzePromptResponse,
    AnalyzeTrackResponse,
    Dimension,
    GenreCount,
    DecadeCount,
    Track,
)


PROMPT_ANALYSIS_SYSTEM = """You are a music expert helping to create playlists from a user's music library.

Analyze the user's prompt and suggest appropriate filters (genres, decades and artists) that would help find matching tracks.

Return a JSON object with:
- genres: Array of genre names that match the prompt (e.g., ["Alternative", "Rock", "Indie"])
- decades: Array of decade strings (e.g., ["1990s", "2000s"])
- artists: Array of artist names the user explicitly asks for (e.g., "a Radiohead playlist" -> ["Radiohead"]). Leave empty when the prompt only mentions an artist as a style reference ("sounds like Radiohead")
- reasoning: Brief explanation of why you chose these filters

If the user asks for specific artists, leave genres and decades empty unless the prompt also asks for them, so no tracks by those artists are filtered out.

Be specific about genres and decades. Consider:
- Mood/atmosphere (melancholy, upbeat, energetic)
- Era references (90s, classic, modern)
- Genre keywords (alternative, jazz, electronic)
- Artist style hints

Return ONLY valid JSON, no markdown formatting."""


TRACK_ANALYSIS_SYSTEM = """You are a music expert analyzing a song to identify its distinctive characteristics.

Given a track's title, artist, album, and year, identify 5-7 specific musical dimensions that make this track unique. These dimensions will help the user explore similar music.

For each dimension, provide:
- id: A short identifier (e.g., "mood", "era", "instrumentation")
- label: A specific, evocative label (NOT generic like "the mood" - be specific like "The melancholy, bittersweet mood")
- description: A brief explanation of this dimension

Make dimensions SPECIFIC to this track, not generic. Bad: "The genre". Good: "90s British alternative rock with Britpop influences".

Return a JSON object with:
{
  "dimensions": [
    {"id": "mood", "label": "The melancholy, introspective mood", "description": "..."},
    ...
  ]
}

Return ONLY valid JSON, no markdown formatting."""


def _artist_key(name: str) -> str:
    """Comparable form of an artist name: lowercase, no accents or leading "the", letters and digits only."""
    name = unidecode(name).strip().lower().removeprefix("the ")
    return "".join(ch for ch in name if ch.isalnum())


def _match_artists(requested: list, library_artists: list[str]) -> list[str]:
    """Map artist names from the AI to the library's spelling; drop ones not in the library."""
    by_key = {_artist_key(name): name for name in library_artists}
    matched = []
    for name in requested:
        if not isinstance(name, str) or not name.strip():
            continue
        found = by_key.get(_artist_key(name))
        if not found:
            best = process.extractOne(name, library_artists, scorer=fuzz.WRatio, score_cutoff=92)
            found = best[0] if best else None
        if found and found not in matched:
            matched.append(found)
    return matched


def analyze_prompt(prompt: str) -> AnalyzePromptResponse:
    """Analyze a natural language prompt to suggest filters.

    Args:
        prompt: User's playlist description

    Returns:
        AnalyzePromptResponse with suggested and available filters

    Raises:
        ValueError: If LLM response cannot be parsed
        RuntimeError: If clients are not initialized
    """
    llm_client = get_llm_client()
    media_client = get_current_media_client()

    if not llm_client:
        raise RuntimeError("LLM client not initialized")
    if not media_client:
        raise RuntimeError("Media server not connected")

    # Available filters: the cache is instant; Jellyfin's live stats scan every track
    if library_cache.has_cached_tracks():
        stats = library_cache.get_cached_genre_decade_stats()
    else:
        stats = media_client.get_library_stats()
    available_genres = [GenreCount(**g) for g in stats.get("genres", [])]
    available_decades = [DecadeCount(**d) for d in stats.get("decades", [])]

    # Build prompt with available filter context
    analysis_prompt = f"""User's playlist request: "{prompt}"

Available genres in their library:
{', '.join(f"{g.name} ({g.count})" if g.count else g.name for g in available_genres[:30])}

Available decades in their library:
{', '.join(f"{d.name} ({d.count})" if d.count else d.name for d in available_decades)}

Suggest genres and decades from the available options that best match the user's request."""

    # Call LLM
    response = llm_client.analyze(analysis_prompt, PROMPT_ANALYSIS_SYSTEM)

    # Parse response
    data = llm_client.parse_json_response(response)

    # Filter suggestions to only include available options
    available_genre_names = {g.name for g in available_genres}
    available_decade_names = {d.name for d in available_decades}

    suggested_genres = [
        g for g in data.get("genres", [])
        if g in available_genre_names
    ]
    suggested_decades = [
        d for d in data.get("decades", [])
        if d in available_decade_names
    ]

    suggested_artists = []
    if data.get("artists") and library_cache.has_cached_tracks():
        suggested_artists = _match_artists(data["artists"], library_cache.get_artist_names())

    return AnalyzePromptResponse(
        suggested_genres=suggested_genres,
        suggested_decades=suggested_decades,
        suggested_artists=suggested_artists,
        available_genres=available_genres,
        available_decades=available_decades,
        reasoning=data.get("reasoning", ""),
        token_count=response.total_tokens,
        estimated_cost=response.estimated_cost(),
    )


def analyze_track(track: Track) -> AnalyzeTrackResponse:
    """Analyze a seed track to extract musical dimensions.

    Args:
        track: Track to analyze

    Returns:
        AnalyzeTrackResponse with track and dimensions

    Raises:
        ValueError: If LLM response cannot be parsed
        RuntimeError: If LLM client is not initialized
    """
    llm_client = get_llm_client()

    if not llm_client:
        raise RuntimeError("LLM client not initialized")

    # Build analysis prompt
    analysis_prompt = f"""Analyze this track:
Title: {track.title}
Artist: {track.artist}
Album: {track.album}
Year: {track.year or "Unknown"}
Genres: {", ".join(track.genres) if track.genres else "Unknown"}

Identify 5-7 specific musical dimensions that make this track distinctive."""

    # Call LLM
    response = llm_client.analyze(analysis_prompt, TRACK_ANALYSIS_SYSTEM)

    # Parse response
    data = llm_client.parse_json_response(response)

    dimensions = [
        Dimension(
            id=d.get("id", f"dim_{i}"),
            label=d.get("label", "Unknown dimension"),
            description=d.get("description", ""),
        )
        for i, d in enumerate(data.get("dimensions", []))
    ]

    return AnalyzeTrackResponse(
        track=track,
        dimensions=dimensions,
        token_count=response.total_tokens,
        estimated_cost=response.estimated_cost(),
    )
