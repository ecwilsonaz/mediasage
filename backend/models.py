"""Pydantic models for MediaSage API contracts and internal data structures."""

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def album_key(artist: str, album: str, lower: bool = True) -> str:
    """Build composite key for artist+album lookups."""
    if lower:
        return f"{artist.lower()}|||{album.lower()}"
    return f"{artist}|||{album}"


# =============================================================================
# Core Entities
# =============================================================================


class Track(BaseModel):
    """A music track from the media library."""

    rating_key: str
    title: str
    artist: str
    album: str
    duration_ms: int
    year: int | None = None
    genres: list[str] = []
    art_url: str | None = None
    user_rating: int | None = None  # 0-10 scale; Plex only

    @property
    def duration_formatted(self) -> str:
        """Return duration as M:SS format."""
        minutes = self.duration_ms // 60000
        seconds = (self.duration_ms % 60000) // 1000
        return f"{minutes}:{seconds:02d}"


class Dimension(BaseModel):
    """A musical dimension identified from a seed track."""

    id: str
    label: str
    description: str


class FilterSet(BaseModel):
    """Filters applied to narrow track selection."""

    genres: list[str] = []
    decades: list[str] = []
    track_count: int = 25
    exclude_live: bool = True

    @field_validator("track_count")
    @classmethod
    def validate_track_count(cls, v: int) -> int:
        if v not in [15, 25, 50, 100]:
            raise ValueError("track_count must be 15, 25, 50, or 100")
        return v


class Playlist(BaseModel):
    """A generated playlist with tracks and metadata."""

    name: str
    tracks: list[Track]
    source_prompt: str | None = None
    seed_track_key: str | None = None
    selected_dimensions: list[str] | None = None

    @property
    def duration_total(self) -> int:
        """Total duration in milliseconds."""
        return sum(t.duration_ms for t in self.tracks)

    @property
    def track_count(self) -> int:
        return len(self.tracks)


# =============================================================================
# Configuration Models
# =============================================================================


class PlexConfig(BaseModel):
    """Plex server connection settings."""

    url: str
    token: str
    music_library: str = "Music"


class JellyfinConfig(BaseModel):
    """Jellyfin server connection settings."""

    url: str = ""
    token: str = ""  # API key from Jellyfin Dashboard > API Keys
    music_library: str = "Music"


class LLMConfig(BaseModel):
    """LLM provider settings."""

    provider: Literal["anthropic", "openai", "gemini", "ollama", "custom"]
    api_key: str = ""  # The active provider's key; optional for local providers
    api_keys: dict[str, str] = {}  # Keys saved in Settings, by provider (not from env vars)
    model_analysis: str
    model_generation: str
    smart_generation: bool = False
    # Local provider settings
    ollama_url: str = "http://localhost:11434"
    ollama_context_window: int = 32768  # Detected from model, can be overridden
    custom_url: str = ""
    custom_context_window: int = 32768
    request_timeout: int = 600  # Seconds to wait for an AI response; local models can be slow

    @field_validator("request_timeout")
    @classmethod
    def validate_request_timeout(cls, v: int) -> int:
        if v < 30:
            raise ValueError("Request timeout must be at least 30 seconds")
        return v

    @field_validator("ollama_context_window", "custom_context_window")
    @classmethod
    def validate_context_window(cls, v: int) -> int:
        if v < 512:
            raise ValueError("Context window must be at least 512 tokens")
        if v > 2000000:
            raise ValueError("Context window cannot exceed 2,000,000 tokens")
        return v


class DefaultsConfig(BaseModel):
    """Default values for UI."""

    track_count: int = 25


class AppConfig(BaseModel):
    """Root configuration object."""

    media_server: Literal["plex", "jellyfin"] = "plex"
    plex: PlexConfig
    jellyfin: JellyfinConfig = JellyfinConfig()
    llm: LLMConfig
    defaults: DefaultsConfig = DefaultsConfig()
    library_sync_hours: float = Field(24, ge=0)  # Re-sync the cache once it's this old; 0 = never


# =============================================================================
# API Request/Response Models
# =============================================================================


class GenreCount(BaseModel):
    """Genre with track count."""

    name: str
    count: int | None = None


class ArtistCount(BaseModel):
    """Artist with track count."""

    name: str
    count: int


class DecadeCount(BaseModel):
    """Decade with track count."""

    name: str
    count: int | None = None


class LibraryStatsResponse(BaseModel):
    """Library statistics response."""

    total_tracks: int
    genres: list[GenreCount]
    decades: list[DecadeCount]


class AnalyzePromptRequest(BaseModel):
    """Request to analyze a natural language prompt."""

    prompt: str = Field(..., min_length=1, max_length=2000)


class AnalyzePromptResponse(BaseModel):
    """Response from prompt analysis."""

    suggested_genres: list[str]
    suggested_decades: list[str]
    suggested_artists: list[str] = []  # Artists the prompt names that are in the library
    available_genres: list[GenreCount]
    available_decades: list[DecadeCount]
    reasoning: str
    token_count: int = 0
    estimated_cost: float = 0.0


class AnalyzeTrackRequest(BaseModel):
    """Request to analyze a seed track for dimensions."""

    rating_key: str


class AnalyzeTrackResponse(BaseModel):
    """Response from track analysis."""

    track: Track
    dimensions: list[Dimension]
    token_count: int = 0
    estimated_cost: float = 0.0


class FilterPreviewRequest(BaseModel):
    """Request to preview filter results."""

    genres: list[str] = []
    decades: list[str] = []
    artists: list[str] = []  # Only these artists
    exclude_artists: list[str] = []
    track_count: int = 25
    max_tracks_to_ai: int = 500  # 0 = no limit
    min_rating: int = Field(0, ge=0, le=10)  # Plex 0-10 scale (half star = 1); 0 = any
    exclude_live: bool = True


class FilterPreviewResponse(BaseModel):
    """Response with filter preview stats."""

    matching_tracks: int  # -1 if unknown
    tracks_to_send: int  # How many will actually be sent to AI
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost: float


class SeedTrackInput(BaseModel):
    """Seed track input for generation."""

    rating_key: str
    selected_dimensions: list[str]


class GenerateRequest(BaseModel):
    """Request to generate a playlist."""

    prompt: str | None = None
    seed_track: SeedTrackInput | None = None
    additional_notes: str | None = None
    refinement_answers: list[str | None] | None = None
    genres: list[str]
    decades: list[str]
    artists: list[str] = []  # Only these artists
    exclude_artists: list[str] = []
    track_count: int = 25
    exclude_live: bool = True
    min_rating: int = Field(0, ge=0, le=10)  # Plex 0-10 scale (half star = 1); 0 = any
    max_tracks_to_ai: int = 500  # 0 = no limit

    @model_validator(mode="after")
    def check_flow(self) -> "GenerateRequest":
        if not self.prompt and not self.seed_track:
            raise ValueError("Either prompt or seed_track must be provided")
        return self


class GenerateResponse(BaseModel):
    """Response from playlist generation."""

    tracks: list[Track]
    token_count: int
    estimated_cost: float
    # Curator narrative fields
    playlist_title: str = ""
    narrative: str = ""
    track_reasons: dict[str, str] = {}


# Plex rating keys are numeric; Jellyfin item IDs are 32 hex characters
_ITEM_ID_RE = re.compile(r"^(\d+|[0-9a-f]{32})$")


def _validate_rating_keys(v: list[str]) -> list[str]:
    """Validate a non-empty list of Plex or Jellyfin item IDs."""
    if not v:
        raise ValueError("At least one track is required")
    for key in v:
        if not _ITEM_ID_RE.match(key):
            raise ValueError(f"Invalid rating key: {key}")
    return v


def _truncate_description(v: str) -> str:
    """Truncate description to 2000 chars for Plex compatibility."""
    return v[:2000] if v else v


class SavePlaylistRequest(BaseModel):
    """Request to save a playlist to Plex."""

    name: str
    rating_keys: list[str]
    description: str = ""  # Playlist description (narrative) saved to Plex

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Playlist name cannot be empty")
        return v.strip()

    @field_validator("description")
    @classmethod
    def truncate_description(cls, v: str) -> str:
        return _truncate_description(v)

    @field_validator("rating_keys")
    @classmethod
    def validate_rating_keys(cls, v: list[str]) -> list[str]:
        return _validate_rating_keys(v)


class SavePlaylistResponse(BaseModel):
    """Response from saving a playlist."""

    success: bool
    playlist_id: str | None = None
    playlist_url: str | None = None
    error: str | None = None
    tracks_added: int | None = None
    tracks_skipped: int | None = None


# =============================================================================
# Instant Queue Models (005)
# =============================================================================


class PlexPlaylistInfo(BaseModel):
    """Lightweight playlist info for the picker."""

    rating_key: str
    title: str
    track_count: int


class PlexClientInfo(BaseModel):
    """Online Plex client info."""

    client_id: str
    name: str
    product: str
    platform: str
    is_playing: bool
    is_mobile: bool = False


class UpdatePlaylistRequest(BaseModel):
    """Request to update an existing playlist."""

    playlist_id: str
    rating_keys: list[str]
    mode: Literal["replace", "append"]
    description: str = ""

    @field_validator("description")
    @classmethod
    def truncate_description(cls, v: str) -> str:
        return _truncate_description(v)

    @field_validator("playlist_id")
    @classmethod
    def validate_playlist_id(cls, v: str) -> str:
        if v != "__scratch__" and not _ITEM_ID_RE.match(v):
            raise ValueError("playlist_id must be '__scratch__' or a Plex/Jellyfin ID")
        return v

    @field_validator("rating_keys")
    @classmethod
    def validate_rating_keys(cls, v: list[str]) -> list[str]:
        return _validate_rating_keys(v)


class UpdatePlaylistResponse(BaseModel):
    """Response from updating a playlist."""

    success: bool
    tracks_added: int = 0
    tracks_skipped: int = 0
    duplicates_skipped: int = 0
    playlist_url: str | None = None
    warning: str | None = None
    error: str | None = None


class PlayQueueRequest(BaseModel):
    """Request to create a play queue."""

    rating_keys: list[str]
    client_id: str
    mode: Literal["replace", "play_next"] = "replace"

    @field_validator("client_id")
    @classmethod
    def validate_client_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("client_id cannot be empty")
        return v

    @field_validator("rating_keys")
    @classmethod
    def validate_rating_keys(cls, v: list[str]) -> list[str]:
        return _validate_rating_keys(v)


class PlayQueueResponse(BaseModel):
    """Response from play queue creation."""

    success: bool
    client_name: str | None = None
    client_product: str | None = None
    tracks_queued: int = 0
    tracks_skipped: int = 0
    error: str | None = None


class ConfigResponse(BaseModel):
    """Config without secrets for display."""

    version: str
    media_server: str = "plex"
    plex_url: str
    plex_connected: bool  # The active media server, Plex or Jellyfin
    plex_token_set: bool  # True if token is configured (without revealing it)
    music_library: str | None
    # Jellyfin fields
    jellyfin_url: str = ""
    jellyfin_token_set: bool = False
    jellyfin_music_library: str = "Music"
    llm_provider: str
    llm_configured: bool
    llm_api_key_set: bool  # True if API key is configured (without revealing it)
    api_key_providers: list[str] = []  # Providers with a key saved or set by environment variable
    model_analysis: str  # The analysis model being used
    model_generation: str  # The generation model being used
    max_tracks_to_ai: int  # Recommended max tracks for this model
    max_albums_to_ai: int  # Recommended max albums for this model
    cost_per_million_input: float  # Cost per million input tokens for generation model
    cost_per_million_output: float  # Cost per million output tokens for generation model
    analysis_cost_per_million_input: float = 0.0  # Cost per million input tokens for analysis model
    analysis_cost_per_million_output: float = 0.0  # Cost per million output tokens for analysis model
    defaults: DefaultsConfig
    # Local provider fields
    ollama_url: str = "http://localhost:11434"
    ollama_context_window: int = 32768
    custom_url: str = ""
    custom_context_window: int = 32768
    request_timeout: int = 600
    library_sync_hours: float = 24
    is_local_provider: bool = False
    # Settings fields set by environment variables (which override Settings), e.g. {"plex_url": "PLEX_URL"}
    env_overrides: dict[str, str] = {}
    # Generation model's long-context pricing, if any
    long_context_threshold: int | None = None
    long_context_cost_per_million_input: float | None = None
    long_context_cost_per_million_output: float | None = None


class UpdateConfigRequest(BaseModel):
    """Partial config update."""

    media_server: str | None = None
    plex_url: str | None = None
    plex_token: str | None = None
    music_library: str | None = None
    # Jellyfin fields
    jellyfin_url: str | None = None
    jellyfin_token: str | None = None
    jellyfin_music_library: str | None = None
    llm_provider: str | None = None
    llm_api_key: str | None = None
    model_analysis: str | None = None
    model_generation: str | None = None
    # Local provider fields
    ollama_url: str | None = None
    ollama_context_window: int | None = None
    custom_url: str | None = None
    custom_context_window: int | None = None
    request_timeout: int | None = None
    library_sync_hours: float | None = Field(None, ge=0)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    plex_connected: bool
    llm_configured: bool
    media_server: str = "plex"


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None


# =============================================================================
# Ollama API Models
# =============================================================================


class OllamaModel(BaseModel):
    """A model available in Ollama."""

    name: str
    size: int = 0
    modified_at: str = ""


class CloudModel(BaseModel):
    """A cloud model offered in Settings."""

    id: str
    label: str
    context_window: int
    max_tracks: int
    cost_per_million_input: float
    cost_per_million_output: float
    legacy: bool = False


class CloudProviderModels(BaseModel):
    """Models offered for one cloud provider, with its defaults."""

    models: list[CloudModel]
    default_analysis: str
    default_generation: str


class CloudModelsResponse(BaseModel):
    """Cloud models keyed by provider."""

    providers: dict[str, CloudProviderModels]


class OllamaModelInfo(BaseModel):
    """Detailed info about an Ollama model."""

    name: str
    context_window: int
    context_detected: bool = True  # False if using fallback default
    parameter_size: str | None = None


class OllamaModelsResponse(BaseModel):
    """Response from listing Ollama models."""

    models: list[OllamaModel] = []
    error: str | None = None


class OllamaStatus(BaseModel):
    """Connection status for Ollama."""

    connected: bool
    model_count: int = 0
    error: str | None = None


# =============================================================================
# Library Cache Models
# =============================================================================


class SyncProgress(BaseModel):
    """Progress details when sync is running."""

    phase: str | None = None  # "fetching_albums", "fetching", or "processing"
    current: int
    total: int


class LibraryCacheStatusResponse(BaseModel):
    """Response from GET /api/library/status."""

    track_count: int
    synced_at: str | None = None
    is_syncing: bool
    sync_progress: SyncProgress | None = None
    error: str | None = None
    plex_connected: bool
    needs_resync: bool = False
    # For the retry loop and external health checks
    sync_kind: str | None = None          # "full" or "incremental" (last attempt)
    last_attempt_at: str | None = None    # last sync start in this process
    error_at: str | None = None
    incremental_at: str | None = None     # watermark: tracks changed after this are picked up next time
    interrupted: bool = False             # a full sync was running when the process died (re-sync pending)


class SyncTriggerResponse(BaseModel):
    """Response from POST /api/library/sync."""

    started: bool
    blocking: bool = False


# =============================================================================
# Recommendation Models (006)
# =============================================================================


class AlbumCandidate(BaseModel):
    """An album from the user's Plex library, aggregated from cached tracks."""

    parent_rating_key: str
    album: str
    album_artist: str
    year: int | None = None
    genres: list[str] = []
    decade: str = ""
    track_count: int = 0
    track_rating_keys: list[str] = []


class ClarifyingQuestion(BaseModel):
    """A question generated by the LLM to refine the recommendation."""

    question_text: str
    options: list[str]
    dimension: str


class SommelierPitch(BaseModel):
    """The editorial writeup for a recommendation."""

    hook: str = ""
    context: str = ""
    listening_guide: str = ""
    connection: str = ""
    short_pitch: str = ""
    full_text: str = ""


class AlbumRecommendation(BaseModel):
    """The output of the recommendation pipeline."""

    rank: str  # "primary" or "secondary"
    album: str
    artist: str
    year: int | None = None
    rating_key: str | None = None
    track_rating_keys: list[str] = []
    art_url: str | None = None
    pitch: SommelierPitch = SommelierPitch()
    research_available: bool = False


class ResearchData(BaseModel):
    """External research fetched for grounding the pitch."""

    musicbrainz_id: str | None = None
    release_date: str | None = None
    label: str | None = None
    track_listing: list[str] = []
    credits: dict[str, str] = {}
    genre_tags: list[str] = []
    wikipedia_summary: str | None = None
    review_links: list[str] = []
    review_texts: list[str] = []
    cover_art_url: str | None = None
    earliest_release_mbid: str | None = None


class ExtractedFacts(BaseModel):
    """Structured facts extracted from research sources by LLM."""

    origin_story: str = ""
    personnel: list[str] = []
    musical_style: str = ""
    vocal_approach: str = ""
    cultural_context: str = ""
    track_highlights: str = ""
    common_misconceptions: str = ""
    source_coverage: str = ""
    track_listing: list[str] = []  # Authoritative list from MusicBrainz, not LLM-extracted

    def to_text(self, include_track_listing: bool = True) -> str:
        """Format facts as labeled text block for LLM prompts."""
        parts = []
        for label, value in [
            ("Origin", self.origin_story),
            ("Personnel", ", ".join(self.personnel) if self.personnel else ""),
            ("Musical style", self.musical_style),
            ("Vocal approach", self.vocal_approach),
            ("Cultural context", self.cultural_context),
            ("Track highlights", self.track_highlights),
            ("Common misconceptions", self.common_misconceptions),
            ("Source coverage", self.source_coverage),
        ]:
            if value:
                parts.append(f"- {label}: {value}")
        if include_track_listing and self.track_listing:
            parts.append("- Track listing: " + ", ".join(self.track_listing))
        return "\n".join(parts)


class PitchIssue(BaseModel):
    """A factual issue found during pitch validation."""

    claim: str
    problem: str
    correction: str


class PitchValidation(BaseModel):
    """Result of validating a pitch against research data."""

    valid: bool
    issues: list[PitchIssue] = []


class TasteProfile(BaseModel):
    """Aggregate view of the user's library for discovery mode."""

    genre_distribution: dict[str, int] = {}
    decade_distribution: dict[str, int] = {}
    top_artists: list[str] = []
    total_albums: int = 0
    owned_albums: list[dict[str, str]] = []


class RecommendSessionState(BaseModel):
    """Transient state maintained during a recommendation session."""

    mode: Literal["library", "discovery"] = "library"
    prompt: str = ""
    filters: dict[str, list[str]] = {}
    questions: list[ClarifyingQuestion] = []
    answers: list[str | None] = []
    answer_texts: list[str] = []
    album_candidates: list[AlbumCandidate] = []
    taste_profile: TasteProfile | None = None
    familiarity_pref: Literal["any", "comfort", "rediscover", "hidden_gems"] = "any"
    previously_recommended: list[str] = []  # "artist|||album" keys shown in prior rounds
    # Cost accumulators (reset each generation round)
    total_tokens: int = 0
    total_cost: float = 0.0


class AnalyzePromptFiltersRequest(BaseModel):
    """Request to analyze a prompt and suggest genre/decade filters."""

    prompt: str
    genres: list[str] = []
    decades: list[str] = []


class AnalyzePromptFiltersResponse(BaseModel):
    """Response with suggested genre/decade pre-selections."""

    genres: list[str] = []
    decades: list[str] = []
    reasoning: str = ""


class RecommendQuestionsRequest(BaseModel):
    """Request to generate clarifying questions."""

    prompt: str = Field(..., min_length=1, max_length=2000)


class RecommendQuestionsResponse(BaseModel):
    """Response with clarifying questions."""

    questions: list[ClarifyingQuestion]
    session_id: str
    token_count: int = 0
    estimated_cost: float = 0.0


class RecommendSwitchModeRequest(BaseModel):
    """Request to switch a recommendation session to a different mode."""

    session_id: str
    mode: Literal["library", "discovery"]


class RecommendSwitchModeResponse(BaseModel):
    """Response after switching recommendation mode."""

    session_id: str


class RecommendGenerateRequest(BaseModel):
    """Request to generate album recommendations."""

    session_id: str
    answers: list[str | None]
    answer_texts: list[str] = []
    mode: Literal["library", "discovery"] = "library"
    genres: list[str] = []
    decades: list[str] = []
    familiarity_pref: Literal["any", "comfort", "rediscover", "hidden_gems"] = "any"
    max_albums: int = 2500

    @field_validator("max_albums")
    @classmethod
    def validate_max_albums(cls, v: int) -> int:
        if v < 0:
            raise ValueError("max_albums must be non-negative")
        return min(v, 50000)


class RecommendGenerateResponse(BaseModel):
    """Response with album recommendations."""

    recommendations: list[AlbumRecommendation]
    token_count: int = 0
    estimated_cost: float = 0.0
    research_warning: str | None = None


class AlbumPreviewResponse(BaseModel):
    """Response from album preview endpoint."""

    matching_albums: int
    albums_to_send: int
    estimated_input_tokens: int = 0
    estimated_cost: float = 0.0


# =============================================================================
# Results Persistence Models
# =============================================================================


class ResultListItem(BaseModel):
    """A saved result summary for history list (no snapshot)."""

    id: str
    type: str
    title: str
    prompt: str
    track_count: int
    artist: str | None = None
    art_rating_key: str | None = None
    subtitle: str | None = None
    created_at: str


class ResultListResponse(BaseModel):
    """Paginated list of saved results."""

    results: list[ResultListItem]
    total: int


class ResultDetail(BaseModel):
    """Full saved result including snapshot for rendering."""

    id: str
    type: str
    title: str
    prompt: str
    track_count: int
    artist: str | None = None
    art_rating_key: str | None = None
    subtitle: str | None = None
    created_at: str
    snapshot: dict


# =============================================================================
# Setup/Onboarding Models
# =============================================================================


class SetupStatusResponse(BaseModel):
    """Full onboarding checklist state."""

    data_dir_writable: bool
    process_uid: int = 0
    process_gid: int = 0
    data_dir: str = ""
    media_server: str = "plex"
    plex_connected: bool
    plex_error: str | None = None
    plex_from_env: bool = False
    jellyfin_connected: bool = False
    jellyfin_error: str | None = None
    jellyfin_from_env: bool = False
    music_libraries: list[str] = []
    llm_configured: bool
    llm_provider: str = ""
    llm_from_env: bool = False
    library_synced: bool
    track_count: int = 0
    is_syncing: bool = False
    sync_progress: SyncProgress | None = None
    setup_complete: bool


class ValidatePlexRequest(BaseModel):
    """Request to validate Plex credentials during setup."""

    plex_url: str
    plex_token: str
    music_library: str = "Music"


class ValidatePlexResponse(BaseModel):
    """Response from Plex validation."""

    success: bool
    error: str | None = None
    server_name: str | None = None
    music_libraries: list[str] = []


class ValidateJellyfinRequest(BaseModel):
    """Request to validate Jellyfin credentials during setup."""

    jellyfin_url: str
    jellyfin_token: str
    music_library: str = "Music"


class ValidateJellyfinResponse(BaseModel):
    """Response from Jellyfin validation."""

    success: bool
    error: str | None = None
    server_name: str | None = None
    music_libraries: list[str] = []
    user_id: str | None = None


class ValidateAIRequest(BaseModel):
    """Request to validate AI provider credentials during setup."""

    provider: str
    api_key: str = ""
    ollama_url: str = ""
    custom_url: str = ""
    custom_model: str = ""
    ollama_model: str = ""


class ValidateAIResponse(BaseModel):
    """Response from AI provider validation."""

    success: bool
    error: str | None = None
    provider_name: str = ""


class SetupCompleteResponse(BaseModel):
    """Response from marking setup as complete."""

    success: bool
