"""Album recommendation pipeline for MediaSage.

Implements the 4-call LLM pipeline: gap analysis, question generation,
album selection, and pitch writing. Maintains in-memory session state
for "Show me another" functionality.
"""

import logging
import threading
import time
import uuid
from typing import Any

from rapidfuzz import fuzz

from backend.llm_client import LLMClient, LLMResponse
from backend.models import (
    AlbumCandidate,
    AlbumRecommendation,
    ClarifyingQuestion,
    ExtractedFacts,
    PitchIssue,
    PitchValidation,
    RecommendSessionState,
    ResearchData,
    SommelierPitch,
    TasteProfile,
    album_key,
)
from backend.plex_client import simplify_string

# Fuzzy-match thresholds for album selection (0-100 scale, rapidfuzz)
ALBUM_ARTIST_MIN_SCORE = 70   # Minimum artist similarity to consider a match
ALBUM_COMBINED_MIN_SCORE = 70  # Minimum (artist+album)/2 to accept a match

logger = logging.getLogger(__name__)
cost_logger = logging.getLogger("recommend.cost")

# Dimension library for gap analysis
DIMENSION_LIBRARY = [
    {"id": "energy", "label": "Energy Level", "description": "Calm vs intense, quiet vs loud"},
    {"id": "emotional_direction", "label": "Emotional Direction", "description": "Sad, joyful, bittersweet, cathartic, neutral"},
    {"id": "attention_level", "label": "Attention Level", "description": "Background listening vs active listening"},
    {"id": "era", "label": "Era / Time Period", "description": "Classic, contemporary, timeless"},
    {"id": "familiarity", "label": "Familiarity", "description": "Well-known vs deep cuts, mainstream vs obscure"},
    {"id": "vocal_presence", "label": "Vocal Presence", "description": "Instrumental, minimal vocals, vocal-forward"},
    {"id": "lyrical_mood", "label": "Lyrical Mood", "description": "Introspective, storytelling, abstract, anthemic"},
    {"id": "social_context", "label": "Social Context", "description": "Solo listening, with friends, romantic, communal"},
    {"id": "complexity", "label": "Musical Complexity", "description": "Simple and direct vs layered and complex"},
    {"id": "rawness", "label": "Production Style", "description": "Lo-fi/raw vs polished/produced"},
    {"id": "tempo", "label": "Tempo", "description": "Slow, mid-tempo, fast-paced"},
    {"id": "cultural_specificity", "label": "Cultural Specificity", "description": "Universal appeal vs culturally rooted"},
]

# Session expiry in seconds (30 minutes)
SESSION_EXPIRY = 1800
MAX_SESSIONS = 100

# ── Familiarity directives ─────────────────────────────────────────────────

FAMILIARITY_SELECTION_DIRECTIVES = {
    "comfort": (
        "\n\nFAMILIARITY PREFERENCE: The user wants comfort picks. "
        "Strongly prefer albums marked {well-loved}. Avoid {unplayed} albums."
    ),
    "rediscover": (
        "\n\nFAMILIARITY PREFERENCE: The user wants to rediscover forgotten albums. "
        "Strongly prefer albums marked {light}, especially those not played recently. "
        "Avoid {unplayed} albums."
    ),
    "hidden_gems": (
        "\n\nFAMILIARITY PREFERENCE: The user wants hidden gems they haven't explored. "
        "Strongly prefer albums marked {unplayed}. Avoid {well-loved} albums."
    ),
}

FAMILIARITY_PITCH_GUIDANCE = {
    "comfort": (
        "\n\nFamiliarity framing: The user wants comfort picks — albums they already love. "
        "Frame pitches as celebrating a favorite: remind them why they love it, "
        "suggest a fresh angle to appreciate it anew.\n"
    ),
    "rediscover": (
        "\n\nFamiliarity framing: The user wants to rediscover forgotten albums. "
        "Frame pitches as 'when's the last time you sat down with this?' — "
        "highlight what they'll notice on a return visit.\n"
    ),
    "hidden_gems": (
        "\n\nFamiliarity framing: The user wants hidden gems they haven't explored. "
        "Frame pitches as exciting discovery: 'you haven't given this a real shot yet' — "
        "emphasize what makes it worth a dedicated listen.\n"
    ),
}


def format_answers_for_selection(
    answers: list[str | None], answer_texts: list[str]
) -> str:
    """Format answers with Q-labels for album selection prompts."""
    parts = []
    for i, ans in enumerate(answers):
        if ans:
            text = ans
            if i < len(answer_texts) and answer_texts[i]:
                text += f" (also: {answer_texts[i]})"
            parts.append(f"Q{i+1} answer: {text}")
        else:
            parts.append(f"Q{i+1}: skipped")
    return "\n".join(parts)


def format_answers_for_pitch(
    answers: list[str | None], answer_texts: list[str]
) -> str:
    """Format answers as compact semicolon-joined string for pitch prompts."""
    parts = []
    for i, ans in enumerate(answers):
        if ans:
            text = ans
            if i < len(answer_texts) and answer_texts[i]:
                text += f" ({answer_texts[i]})"
            parts.append(text)
    return "; ".join(parts) if parts else "no specific preferences"


class RecommendationPipeline:
    """Orchestrates the album recommendation flow."""

    def __init__(self, config: Any, llm_client: LLMClient):
        self.config = config
        self.llm_client = llm_client
        self._sessions: dict[str, tuple[RecommendSessionState, float]] = {}
        self._session_lock = threading.Lock()

    def _log_cost(
        self,
        call_name: str,
        response: LLMResponse,
        session_id: str,
        album_count: int = 0,
    ) -> None:
        """Emit structured cost log line and accumulate session totals."""
        cost = response.estimated_cost()
        tokens = response.input_tokens + response.output_tokens
        cost_logger.info(
            "recommend.cost | call=%s model=%s input=%d output=%d cost=%.5f albums=%d session=%s",
            call_name,
            response.model,
            response.input_tokens,
            response.output_tokens,
            cost,
            album_count,
            session_id,
        )
        # Accumulate into session for response payload
        with self._session_lock:
            entry = self._sessions.get(session_id)
            if entry:
                session_state, ts = entry
                session_state.total_tokens += tokens
                session_state.total_cost += cost

    # ── Fact extraction ─────────────────────────────────────────────────

    def extract_facts(
        self,
        artist: str,
        album: str,
        research: ResearchData,
        session_id: str,
    ) -> ExtractedFacts:
        """Extract structured facts from raw research data.

        Uses the generation (cheap) model to convert raw Wikipedia text,
        review content, and MusicBrainz data into labeled, structured facts.

        Returns ExtractedFacts with fields populated from sources.
        """
        # Build source material
        sources = []

        if research.wikipedia_summary:
            sources.append(f"WIKIPEDIA:\n{research.wikipedia_summary}")

        for i, review in enumerate(research.review_texts):
            sources.append(f"REVIEW {i + 1}:\n{review}")

        if research.track_listing:
            tracks = ", ".join(research.track_listing)
            sources.append(f"TRACK LISTING:\n{tracks}")

        metadata_parts = []
        if research.release_date:
            metadata_parts.append(f"Release date: {research.release_date}")
        if research.label:
            metadata_parts.append(f"Label: {research.label}")
        if research.credits:
            creds = ", ".join(f"{role}: {name}" for role, name in research.credits.items())
            metadata_parts.append(f"Credits: {creds}")
        if metadata_parts:
            sources.append("MUSICBRAINZ METADATA:\n" + "\n".join(metadata_parts))

        sources_text = "\n\n".join(sources) if sources else "No sources available."

        system = (
            "You are a music research assistant. Extract verifiable facts about a specific "
            "album from the provided sources. Follow these rules strictly:\n\n"
            "1. ONLY state facts that appear in the sources below. Do not add knowledge from "
            "your training data.\n"
            "2. If a topic is not covered in the sources, write \"NOT IN SOURCES\" for that field.\n"
            "3. If sources conflict on a point, note the conflict.\n"
            "4. Be specific to THIS album — do not generalize from the artist's broader catalog.\n"
            "5. For vocal_approach, note the specific language(s) used and whether it varies by track.\n"
            "6. For common_misconceptions, note anything the sources clarify that could easily be "
            "misunderstood or overgeneralized.\n\n"
            "Return a JSON object with these fields:\n"
            "- origin_story: How/why the album was made, key events in its creation\n"
            "- personnel: List of key people involved (musicians, producers, engineers)\n"
            "- musical_style: Sound, instrumentation, production approach\n"
            "- vocal_approach: Language(s) sung in, singing style, notable vocal choices\n"
            "- cultural_context: Reception, significance, scene/movement\n"
            "- track_highlights: Notable individual tracks mentioned in sources\n"
            "- common_misconceptions: Things sources clarify or correct about common assumptions\n"
            "- source_coverage: Brief note on what topics the sources cover well vs poorly\n\n"
            "No explanation, just the JSON object."
        )

        user_prompt = (
            f"Album: {artist} — {album}\n\n"
            f"SOURCES:\n{sources_text}\n\n"
            f"Extract the structured facts."
        )

        response = self.llm_client.generate(user_prompt, system)
        self._log_cost("fact_extraction", response, session_id)

        raw = self.llm_client.parse_json_response(response)
        return ExtractedFacts(
            origin_story=raw.get("origin_story", ""),
            personnel=raw.get("personnel", []),
            musical_style=raw.get("musical_style", ""),
            vocal_approach=raw.get("vocal_approach", ""),
            cultural_context=raw.get("cultural_context", ""),
            track_highlights=raw.get("track_highlights", ""),
            common_misconceptions=raw.get("common_misconceptions", ""),
            source_coverage=raw.get("source_coverage", ""),
            track_listing=research.track_listing,  # Authoritative MusicBrainz data
        )

    # ── Session management ──────────────────────────────────────────────

    def create_session(self, session_state: RecommendSessionState) -> str:
        """Create a new recommendation session, return session_id."""
        with self._session_lock:
            self._expire_old_sessions()
            session_id = f"rec_{uuid.uuid4().hex[:12]}"
            self._sessions[session_id] = (session_state, time.time())
            return session_id

    def migrate_sessions_from(self, other: "RecommendationPipeline") -> None:
        """Thread-safe session migration from another pipeline instance."""
        with other._session_lock:
            with self._session_lock:
                self._sessions = dict(other._sessions)

    def get_session(self, session_id: str) -> RecommendSessionState | None:
        """Retrieve a session by ID, or None if expired/missing."""
        with self._session_lock:
            self._expire_old_sessions()
            entry = self._sessions.get(session_id)
            if entry is None:
                return None
            session_state, created_at = entry
            # Touch timestamp on access
            self._sessions[session_id] = (session_state, time.time())
            return session_state

    def update_session_questions(self, session_id: str, questions: list) -> None:
        """Thread-safe update of session questions (after generation)."""
        with self._session_lock:
            entry = self._sessions.get(session_id)
            if entry:
                session_state, _ = entry
                session_state.questions = questions
                self._sessions[session_id] = (session_state, time.time())

    def update_session_answers(
        self, session_id: str, answers: list, answer_texts: list
    ) -> None:
        """Thread-safe update of session answers."""
        with self._session_lock:
            entry = self._sessions.get(session_id)
            if entry:
                session_state, _ = entry
                session_state.answers = answers
                session_state.answer_texts = answer_texts
                self._sessions[session_id] = (session_state, time.time())

    def update_session_generate_state(
        self,
        session_id: str,
        mode: str,
        filters: dict,
        familiarity_pref: str,
        album_candidates: list[AlbumCandidate] | None = None,
        taste_profile: TasteProfile | None = None,
    ) -> None:
        """Thread-safe update of session state for generate requests."""
        with self._session_lock:
            entry = self._sessions.get(session_id)
            if entry:
                session_state, _ = entry
                session_state.mode = mode
                session_state.filters = filters
                session_state.familiarity_pref = familiarity_pref
                if album_candidates is not None:
                    session_state.album_candidates = album_candidates
                if taste_profile is not None:
                    session_state.taste_profile = taste_profile
                # Reset cost accumulators for this generation round
                session_state.total_tokens = 0
                session_state.total_cost = 0.0
                self._sessions[session_id] = (session_state, time.time())

    def get_session_costs(self, session_id: str) -> tuple[int, float]:
        """Read accumulated token count and cost for the current generation round."""
        with self._session_lock:
            entry = self._sessions.get(session_id)
            if entry:
                session_state, _ = entry
                return session_state.total_tokens, session_state.total_cost
        return 0, 0.0

    def update_previously_recommended(
        self, session_id: str, new_keys: list[str]
    ) -> None:
        """Thread-safe update of previously recommended albums."""
        with self._session_lock:
            entry = self._sessions.get(session_id)
            if entry:
                session_state, _ = entry
                for key in new_keys:
                    if key not in session_state.previously_recommended:
                        session_state.previously_recommended.append(key)
                session_state.previously_recommended = (
                    session_state.previously_recommended[-30:]
                )
                self._sessions[session_id] = (session_state, time.time())

    def delete_session(self, session_id: str) -> None:
        """Remove a session by ID."""
        with self._session_lock:
            self._sessions.pop(session_id, None)

    def _expire_old_sessions(self) -> None:
        """Remove expired sessions and evict oldest if over MAX_SESSIONS. Caller must hold _session_lock."""
        now = time.time()
        expired = [
            sid for sid, (_, ts) in self._sessions.items()
            if now - ts > SESSION_EXPIRY
        ]
        for sid in expired:
            del self._sessions[sid]
            logger.info("Expired recommendation session %s", sid)
        # Evict oldest sessions if over the cap
        while len(self._sessions) > MAX_SESSIONS:
            oldest_sid = min(self._sessions, key=lambda s: self._sessions[s][1])
            del self._sessions[oldest_sid]
            logger.info("Evicted oldest recommendation session %s (over cap)", oldest_sid)

    # ── Pipeline steps ──────────────────────────────────────────────────

    def gap_analysis(self, prompt: str, session_id: str) -> list[str]:
        """Identify the 2 most impactful dimensions to clarify.

        Returns list of 2 dimension IDs.
        """
        dimension_text = "\n".join(
            f"- {d['id']}: {d['label']} — {d['description']}"
            for d in DIMENSION_LIBRARY
        )

        system = (
            "You are a music taste analyst. Given a user's album recommendation prompt, "
            "identify which 2 musical dimensions from the provided list would most help "
            "narrow down the perfect album. Return ONLY a JSON array of exactly 2 dimension "
            "IDs, e.g. [\"energy\", \"emotional_direction\"]. No explanation."
        )

        user_prompt = (
            f"User wants: \"{prompt}\"\n\n"
            f"Available dimensions:\n{dimension_text}\n\n"
            f"Which 2 dimensions have the biggest gap — where knowing the user's preference "
            f"would most change which album you'd recommend? Return JSON array of 2 IDs."
        )

        response = self.llm_client.analyze(user_prompt, system)
        self._log_cost("gap_analysis", response, session_id)

        dimensions = self.llm_client.parse_json_response(response)
        if not isinstance(dimensions, list) or len(dimensions) < 2:
            # Fallback to first two dimensions
            return ["energy", "emotional_direction"]

        # Validate dimension IDs
        valid_ids = {d["id"] for d in DIMENSION_LIBRARY}
        result = [d for d in dimensions[:2] if d in valid_ids]
        while len(result) < 2:
            for d in DIMENSION_LIBRARY:
                if d["id"] not in result:
                    result.append(d["id"])
                    break
        return result[:2]

    def analyze_prompt_filters(
        self,
        prompt: str,
        available_genres: list[str],
        available_decades: list[str],
    ) -> dict:
        """Analyze a prompt and suggest relevant genre/decade filters.

        Returns dict with 'genres', 'decades', and 'reasoning' keys.
        Validates against available lists; falls back to all-available if empty.
        """
        system = (
            "You are a music librarian helping pre-select filters for an album recommendation.\n"
            "Given a user's prompt, select which genres and decades from the available lists are RELEVANT.\n"
            "Rules:\n"
            "- Be inclusive but not indiscriminate.\n"
            "- For vague/mood prompts: select broadly but exclude clearly irrelevant genres "
            "(e.g., exclude Holiday for \"dark and heavy\").\n"
            "- For specific genre/era prompts: select narrowly.\n"
            "- When in doubt, include rather than exclude.\n"
            "- If no time period mentioned: select ALL decades.\n"
            "Return JSON: { \"genres\": [...], \"decades\": [...], \"reasoning\": \"...\" }"
        )

        user_prompt = (
            f"User wants: \"{prompt}\"\n\n"
            f"Available genres: {', '.join(available_genres)}\n\n"
            f"Available decades: {', '.join(available_decades)}\n\n"
            f"Which genres and decades are relevant?"
        )

        response = self.llm_client.generate(user_prompt, system)
        self._log_cost("prompt_filter_analysis", response, "n/a")

        raw = self.llm_client.parse_json_response(response)

        # Validate against available lists
        genre_set = set(available_genres)
        decade_set = set(available_decades)
        genres = [g for g in raw.get("genres", []) if g in genre_set]
        decades = [d for d in raw.get("decades", []) if d in decade_set]

        # Fallback: if empty, return all available
        if not genres:
            genres = list(available_genres)
        if not decades:
            decades = list(available_decades)

        return {
            "genres": genres,
            "decades": decades,
            "reasoning": raw.get("reasoning", ""),
            "token_count": response.total_tokens,
            "estimated_cost": response.estimated_cost(),
        }

    def generate_questions(
        self, prompt: str, dimension_ids: list[str], session_id: str
    ) -> list[ClarifyingQuestion]:
        """Generate 2 clarifying questions based on selected dimensions.

        Returns list of ClarifyingQuestion objects.
        """
        dim_lookup = {d["id"]: d for d in DIMENSION_LIBRARY}
        dim_descriptions = []
        for did in dimension_ids:
            d = dim_lookup.get(did, {"label": did, "description": did})
            dim_descriptions.append(f"{d['label']}: {d['description']}")

        system = (
            "You are a friendly music recommendation assistant. Generate exactly 2 clarifying "
            "questions to help pick the perfect album. Each question should:\n"
            "- Reference the user's words naturally\n"
            "- Have 3-4 short, tappable answer options\n"
            "- Address the specified musical dimension\n\n"
            "Return JSON array of objects with: question_text, options (array of 3-4 strings), dimension (the dimension id).\n"
            "No explanation, just the JSON array."
        )

        user_prompt = (
            f"User wants: \"{prompt}\"\n\n"
            f"Dimensions to ask about:\n"
            + "\n".join(f"- {did}: {desc}" for did, desc in zip(dimension_ids, dim_descriptions))
            + "\n\nGenerate 2 natural, conversational questions."
        )

        response = self.llm_client.generate(user_prompt, system)
        self._log_cost("question_gen", response, session_id)

        raw = self.llm_client.parse_json_response(response)
        if not isinstance(raw, list):
            return []
        questions = []
        for item in raw[:2]:
            questions.append(ClarifyingQuestion(
                question_text=item.get("question_text", ""),
                options=item.get("options", [])[:4],
                dimension=item.get("dimension", ""),
            ))
        return questions

    def select_albums(
        self,
        prompt: str,
        answers: list[str | None],
        answer_texts: list[str],
        album_candidates: list[AlbumCandidate],
        session_id: str,
        familiarity_pref: str = "any",
        familiarity_data: dict[str, dict] | None = None,
        previously_recommended: list[str] | None = None,
    ) -> list[AlbumRecommendation]:
        """Select 1 primary + 2 secondary albums from the candidate pool.

        Returns list of AlbumRecommendation (without pitches yet).
        """
        # Filter out previously recommended albums
        if previously_recommended:
            excluded = set(previously_recommended)
            album_candidates = [
                c for c in album_candidates
                if album_key(c.album_artist, c.album) not in excluded
            ]

        # Edge case: very small pool — return all candidates directly
        if len(album_candidates) <= 3:
            recs = []
            for i, c in enumerate(album_candidates):
                recs.append(AlbumRecommendation(
                    rank="primary" if i == 0 else "secondary",
                    album=c.album,
                    artist=c.album_artist,
                    year=c.year,
                    rating_key=c.parent_rating_key,
                    track_rating_keys=c.track_rating_keys,
                    art_url=f"/api/art/{c.track_rating_keys[0]}" if c.track_rating_keys else None,
                ))
            return recs

        # Build album list for LLM
        album_lines = []
        for a in album_candidates:
            genres_str = ", ".join(a.genres[:3]) if a.genres else "Unknown"
            line = f"- {a.album_artist} — {a.album} ({a.year or '?'}) [{genres_str}]"
            if familiarity_pref != "any" and familiarity_data and a.parent_rating_key in familiarity_data:
                level = familiarity_data[a.parent_rating_key]["level"]
                line += f" {{{level}}}"
            album_lines.append(line)
        album_text = "\n".join(album_lines)

        # Build answer context
        answers_text = format_answers_for_selection(answers, answer_texts)
        familiarity_directive = FAMILIARITY_SELECTION_DIRECTIVES.get(familiarity_pref, "")

        system = (
            "You are a music recommendation expert. Pick exactly 3 albums from the provided list "
            "that best match the user's request and clarifying answers. The first pick is the PRIMARY "
            "recommendation (best match), the other two are SECONDARY (worth exploring).\n\n"
            "Return a JSON array of 3 objects, each with: artist (string), album (string), rank "
            "(\"primary\" for first, \"secondary\" for others). Pick from the list EXACTLY as written.\n"
            "No explanation, just the JSON array."
            f"{familiarity_directive}"
        )

        small_pool_note = ""
        if len(album_candidates) < 10:
            small_pool_note = (
                "\nNote: The pool is small. Pick the best matches available, "
                "even if the fit isn't perfect. Do your best with what's here."
            )

        user_prompt = (
            f"User wants: \"{prompt}\"\n\n"
            f"Clarifying answers:\n{answers_text}\n\n"
            f"Available albums ({len(album_candidates)} total):\n{album_text}\n\n"
            f"Pick 3 albums: 1 primary + 2 secondary.{small_pool_note}"
        )

        response = self.llm_client.generate(user_prompt, system)
        self._log_cost("selection", response, session_id, album_count=len(album_candidates))

        raw = self.llm_client.parse_json_response(response)
        recommendations = []
        # Build lookup for matching
        candidate_lookup: dict[str, AlbumCandidate] = {}
        for c in album_candidates:
            candidate_lookup[album_key(c.album_artist, c.album)] = c

        for item in raw[:3]:
            artist = item.get("artist", "")
            album = item.get("album", "")
            rank = item.get("rank", "secondary")

            # Match back to candidate (case-insensitive)
            candidate = candidate_lookup.get(album_key(artist, album))

            # Fallback: substring match handles LLM dropping suffixes like "(Reissue)"
            if candidate is None:
                artist_l = artist.lower()
                album_l = album.lower()
                for ckey, cval in candidate_lookup.items():
                    c_artist, c_album = ckey.split("|||", 1)
                    if c_artist == artist_l and (
                        album_l in c_album or c_album in album_l
                    ):
                        candidate = cval
                        break

            # Fallback: fuzzy match for slight name variations
            if candidate is None:
                best_score = 0
                best_candidate = None
                simplified_artist = simplify_string(artist)
                simplified_album = simplify_string(album)
                for cval in candidate_lookup.values():
                    artist_score = fuzz.ratio(simplified_artist, simplify_string(cval.album_artist))
                    if artist_score < ALBUM_ARTIST_MIN_SCORE:
                        continue
                    album_score = fuzz.ratio(simplified_album, simplify_string(cval.album))
                    combined = (artist_score + album_score) / 2
                    if combined > best_score and combined >= ALBUM_COMBINED_MIN_SCORE:
                        best_score = combined
                        best_candidate = cval
                if best_candidate:
                    candidate = best_candidate
                    logger.info(
                        "Fuzzy matched LLM selection '%s — %s' to '%s — %s' (score: %.0f)",
                        artist, album, candidate.album_artist, candidate.album, best_score,
                    )

            # Skip unmatched albums — unplayable in library mode
            if candidate is None:
                logger.warning(
                    "Skipping unmatched LLM album selection: %s — %s", artist, album,
                )
                continue

            rec = AlbumRecommendation(
                rank=rank if rank in ("primary", "secondary") else "secondary",
                album=candidate.album,
                artist=candidate.album_artist,
                year=candidate.year,
                rating_key=candidate.parent_rating_key,
                track_rating_keys=candidate.track_rating_keys,
                art_url=f"/api/art/{candidate.track_rating_keys[0]}" if candidate.track_rating_keys else None,
            )
            recommendations.append(rec)

        # Ensure we have at least 1 primary
        if recommendations and all(r.rank == "secondary" for r in recommendations):
            recommendations[0].rank = "primary"

        return recommendations

    def write_pitches(
        self,
        recommendations: list[AlbumRecommendation],
        prompt: str,
        answers: list[str | None],
        answer_texts: list[str],
        session_id: str,
        research: dict[str, ResearchData] | None = None,
        familiarity_pref: str = "any",
        familiarity_data: dict[str, dict] | None = None,
        extracted_facts: dict[str, ExtractedFacts] | None = None,
    ) -> list[AlbumRecommendation]:
        """Write sommelier pitches for each recommendation.

        Args:
            familiarity_pref: User's familiarity preference ("any"|"comfort"|"rediscover"|"hidden_gems")
            familiarity_data: Optional dict mapping rating_key -> {"level": str, "last_viewed_at": str|None}
            extracted_facts: Optional dict mapping "artist|||album" -> ExtractedFacts

        Returns the same recommendations with pitches filled in.
        """
        # Build album descriptions for context
        album_descs = []
        for rec in recommendations:
            desc = f"[{rec.rank.upper()}] {rec.artist} — {rec.album} ({rec.year or '?'})"
            # Add familiarity context
            if familiarity_data and rec.rating_key and rec.rating_key in familiarity_data:
                level = familiarity_data[rec.rating_key]["level"]
                desc += f"\nFamiliarity: {level}"

            # Add extracted facts if available (preferred over raw research)
            lookup_key = album_key(rec.artist, rec.album)
            if extracted_facts and lookup_key in extracted_facts:
                ef = extracted_facts[lookup_key]
                facts_text = ef.to_text(include_track_listing=False)
                if facts_text:
                    desc += f"\n\nEXTRACTED FACTS (from Wikipedia, MusicBrainz, and reviews):\n{facts_text}"

            # Add track listing from research if available
            if research and lookup_key in research:
                rd = research[lookup_key]
                if rd.track_listing:
                    desc += f"\n\nTRACK LISTING: {', '.join(rd.track_listing)}"
                if rd.label:
                    desc += f"\nLabel: {rd.label}"
                if rd.release_date:
                    desc += f"\nRelease: {rd.release_date}"

            album_descs.append(desc)

        albums_text = "\n\n".join(album_descs)

        # Build answer context
        answers_str = format_answers_for_pitch(answers, answer_texts)
        familiarity_guidance = FAMILIARITY_PITCH_GUIDANCE.get(familiarity_pref, "")

        grounding_rules = ""
        if extracted_facts:
            grounding_rules = (
                "\n\nGROUNDING RULES (mandatory when EXTRACTED FACTS are provided):\n"
                "- Base all factual claims on the EXTRACTED FACTS provided for each album. "
                "Do not rely on general knowledge about the artist when it conflicts with "
                "album-specific facts.\n"
                "- If a fact is marked \"NOT IN SOURCES,\" do not make claims about that topic. "
                "Write around it or keep the language vague and subjective.\n"
                "- Never generalize from an artist's broader catalog to this specific album. "
                "What is true of an artist in general may not be true of this particular record.\n"
                "- Distinguish between the album's actual story and a plausible-sounding narrative. "
                "If the origin field describes specific events, use those — do not invent "
                "a more dramatic or simplified version.\n"
                "- Pay close attention to the 'Common misconceptions' field — these are facts "
                "that are easily gotten wrong.\n"
            )

        system = (
            "You are a passionate music sommelier. Write compelling pitches for album recommendations.\n\n"
            "For the PRIMARY album, write:\n"
            "- hook: A compelling one-liner that makes someone want to press play immediately\n"
            "- context: An interesting detail about the album (recording story, cultural significance, artist journey)\n"
            "- listening_guide: How to approach the listen — what to expect as it unfolds\n"
            "- connection: Why THIS album matches THIS specific request\n\n"
            "For each SECONDARY album, write:\n"
            "- short_pitch: 2-3 vivid sentences that sell the album\n\n"
            "Use specific, vivid language. Reference the user's words. Avoid generic music-critic clichés.\n"
            f"{grounding_rules}"
            f"{familiarity_guidance}\n"
            "Return JSON array of objects with: artist, album, hook, context, listening_guide, connection "
            "(for primary), or short_pitch (for secondary). Include all applicable fields.\n"
            "No explanation, just the JSON array."
        )

        user_prompt = (
            f"User wanted: \"{prompt}\"\n"
            f"Their preferences: {answers_str}\n\n"
            f"Albums to pitch:\n{albums_text}\n\n"
            f"Write the pitches."
        )

        response = self.llm_client.analyze(user_prompt, system)
        self._log_cost("pitch_writing", response, session_id)

        raw = self.llm_client.parse_json_response(response)

        # Match pitches back to recommendations
        pitch_lookup = {}
        for item in raw:
            pitch_lookup[album_key(item.get("artist", ""), item.get("album", ""))] = item

        for rec in recommendations:
            item = pitch_lookup.get(album_key(rec.artist, rec.album))

            # Fallback: LLMs often drop parenthetical suffixes like "(Reissue)"
            # so try fuzzy matching on album name when artist matches well
            if item is None:
                rec_artist_l = rec.artist.lower()
                rec_album_l = rec.album.lower()
                best_score = 0
                best_val = None
                for pkey, pval in pitch_lookup.items():
                    p_artist, p_album = pkey.split("|||", 1)
                    artist_score = fuzz.ratio(rec_artist_l, p_artist)
                    album_score = fuzz.ratio(rec_album_l, p_album)
                    combined = (artist_score + album_score) / 2
                    if combined > best_score and artist_score >= 80 and album_score >= 60:
                        best_score = combined
                        best_val = pval
                if best_val is not None:
                    item = best_val

            if item is None:
                item = {}

            if rec.rank == "primary":
                hook = item.get("hook", "")
                context = item.get("context", "")
                listening_guide = item.get("listening_guide", "")
                connection = item.get("connection", "")
                parts = [p for p in [hook, context, listening_guide, connection] if p]
                full_text = "\n\n".join(parts)
                rec.pitch = SommelierPitch(
                    hook=hook,
                    context=context,
                    listening_guide=listening_guide,
                    connection=connection,
                    full_text=full_text,
                )
            else:
                short_pitch = item.get("short_pitch", "")
                rec.pitch = SommelierPitch(
                    short_pitch=short_pitch,
                    full_text=short_pitch,
                )

            # Mark research availability
            if research and album_key(rec.artist, rec.album) in research:
                rec.research_available = True

        return recommendations

    def validate_pitch(
        self,
        pitch: SommelierPitch,
        facts: ExtractedFacts,
        session_id: str,
    ) -> PitchValidation:
        """Validate a primary album pitch against extracted facts.

        Uses the analysis (smart) model to fact-check claims.

        Returns PitchValidation with any issues found.
        """
        facts_text = facts.to_text(include_track_listing=False)
        if facts.track_listing:
            facts_text += "\n\nAUTHORITATIVE TRACK LISTING:\n"
            facts_text += "\n".join(f"  - {t}" for t in facts.track_listing)

        system = (
            "You are a fact-checker reviewing an album recommendation pitch against "
            "research data. Flag claims that:\n"
            "1. Contradict the extracted facts\n"
            "2. Are not supported by any source and could be wrong (specific biographical "
            "events, specific recording details, specific personnel claims)\n"
            "3. Overgeneralize from the artist's catalog to this specific album\n"
            "4. Mischaracterize events (e.g., 'toured with' vs 'rehearsed with')\n"
            "5. Reference specific track names that do NOT appear in the AUTHORITATIVE TRACK "
            "LISTING. If the pitch mentions a track by name, it must match a track in the "
            "listing (minor punctuation differences are OK).\n\n"
            "Do NOT flag:\n"
            "- Subjective/editorial language (e.g., 'sonic warm bath', 'ethereal')\n"
            "- Vague statements that don't make specific factual claims\n"
            "- Opinions about how the album sounds or feels\n\n"
            "Return a JSON object: {\"valid\": true} if no issues, or "
            "{\"valid\": false, \"issues\": [{\"claim\": \"...\", \"problem\": \"...\", "
            "\"correction\": \"...\"}]} if issues found.\n"
            "No explanation, just the JSON object."
        )

        user_prompt = (
            f"PITCH TO CHECK:\n{pitch.full_text}\n\n"
            f"EXTRACTED FACTS:\n{facts_text}\n\n"
            f"Are there any factual inaccuracies in the pitch?"
        )

        response = self.llm_client.analyze(user_prompt, system)
        self._log_cost("pitch_validation", response, session_id)

        raw = self.llm_client.parse_json_response(response)
        issues = []
        for issue_data in raw.get("issues", []):
            issues.append(PitchIssue(
                claim=issue_data.get("claim", ""),
                problem=issue_data.get("problem", ""),
                correction=issue_data.get("correction", ""),
            ))

        return PitchValidation(
            valid=raw.get("valid", True),
            issues=issues,
        )

    def rewrite_pitch(
        self,
        rec: AlbumRecommendation,
        facts: ExtractedFacts,
        validation: PitchValidation,
        prompt: str,
        answers_str: str,
        session_id: str,
    ) -> None:
        """Rewrite a primary album pitch incorporating validation corrections.

        Mutates rec.pitch in place with the corrected version.
        """
        # Build corrections block
        corrections = []
        for issue in validation.issues:
            corrections.append(
                f"- WRONG: \"{issue.claim}\" → RIGHT: \"{issue.correction}\""
            )
        corrections_text = "\n".join(corrections)

        facts_text = facts.to_text()

        system = (
            "You are a passionate music sommelier. Rewrite this album pitch, fixing the "
            "factual errors listed below. Keep the same tone, structure, and enthusiasm — "
            "only change the parts that are factually wrong.\n\n"
            "Write:\n"
            "- hook: A compelling one-liner\n"
            "- context: An interesting factual detail about the album\n"
            "- listening_guide: How to approach the listen\n"
            "- connection: Why this album matches the request\n\n"
            "Return a JSON object with: hook, context, listening_guide, connection.\n"
            "No explanation, just the JSON object."
        )

        user_prompt = (
            f"Album: {rec.artist} — {rec.album} ({rec.year or '?'})\n"
            f"User wanted: \"{prompt}\"\n"
            f"Their preferences: {answers_str}\n\n"
            f"CORRECTIONS (do not repeat these errors):\n{corrections_text}\n\n"
            f"EXTRACTED FACTS:\n{facts_text}\n\n"
            f"ORIGINAL PITCH:\n{rec.pitch.full_text}\n\n"
            f"Rewrite the pitch fixing the errors above."
        )

        response = self.llm_client.analyze(user_prompt, system)
        self._log_cost("pitch_rewrite", response, session_id)

        item = self.llm_client.parse_json_response(response)

        hook = item.get("hook", "")
        context = item.get("context", "")
        listening_guide = item.get("listening_guide", "")
        connection = item.get("connection", "")
        parts = [p for p in [hook, context, listening_guide, connection] if p]
        full_text = "\n\n".join(parts)

        rec.pitch = SommelierPitch(
            hook=hook,
            context=context,
            listening_guide=listening_guide,
            connection=connection,
            full_text=full_text,
        )

    # ── Discovery mode helpers ──────────────────────────────────────────

    def build_taste_profile(self, album_candidates: list[AlbumCandidate]) -> TasteProfile:
        """Aggregate full album list into a taste profile for discovery mode."""
        genre_dist: dict[str, int] = {}
        decade_dist: dict[str, int] = {}
        artist_counts: dict[str, int] = {}
        owned: list[dict[str, str]] = []

        for album in album_candidates:
            for genre in album.genres:
                genre_dist[genre] = genre_dist.get(genre, 0) + 1
            if album.decade:
                decade_dist[album.decade] = decade_dist.get(album.decade, 0) + 1
            artist_counts[album.album_artist] = artist_counts.get(album.album_artist, 0) + 1
            owned.append({"artist": album.album_artist, "album": album.album})

        top_artists = sorted(artist_counts, key=artist_counts.get, reverse=True)[:20]

        return TasteProfile(
            genre_distribution=genre_dist,
            decade_distribution=decade_dist,
            top_artists=top_artists,
            total_albums=len(album_candidates),
            owned_albums=owned,
        )

    def select_discovery_albums(
        self,
        prompt: str,
        answers: list[str | None],
        answer_texts: list[str],
        taste_profile: TasteProfile,
        session_id: str,
        previously_recommended: list[str] | None = None,
        max_exclusion_albums: int = 2500,
    ) -> list[AlbumRecommendation]:
        """Select 1 primary + 2 secondary albums NOT in the user's library.

        Uses taste profile as context and owned_albums as exclusion list.
        Returns list of AlbumRecommendation (without pitches or rating_keys).
        """
        # Build taste summary
        top_genres = sorted(taste_profile.genre_distribution, key=taste_profile.genre_distribution.get, reverse=True)[:10]
        top_decades = sorted(taste_profile.decade_distribution, key=taste_profile.decade_distribution.get, reverse=True)[:5]

        taste_text = (
            f"Top genres: {', '.join(top_genres)}\n"
            f"Top decades: {', '.join(top_decades)}\n"
            f"Top artists: {', '.join(taste_profile.top_artists[:10])}\n"
            f"Library size: {taste_profile.total_albums} albums"
        )

        # Build exclusion list — send owned albums so the LLM avoids them.
        # Capped at max_exclusion_albums to stay within context windows.
        # Post-filtering via owned_set catches any the LLM misses.
        exclusion_albums = taste_profile.owned_albums[:max_exclusion_albums]
        exclusion_text = "\n".join(
            f"- {a['artist']} — {a['album']}" for a in exclusion_albums
        )

        # Build answer context
        answers_text = format_answers_for_selection(answers, answer_texts)

        system = (
            "You are a music recommendation expert with encyclopedic knowledge. "
            "Recommend 7 albums the user does NOT already own that match their request and taste profile. "
            "The first pick is the PRIMARY recommendation (best match), the others are SECONDARY.\n\n"
            "IMPORTANT: Do NOT recommend any album from the exclusion list below. "
            "Recommend real, existing albums with correct artist names and years.\n\n"
            "Return a JSON array of 5 objects, each with: artist (string), album (string), "
            "year (integer), rank (\"primary\" for first, \"secondary\" for others).\n"
            "No explanation, just the JSON array."
        )

        # Add previously recommended albums to exclusion
        prev_text = ""
        if previously_recommended:
            prev_lines = []
            for key in previously_recommended:
                parts = key.split("|||")
                if len(parts) == 2:
                    prev_lines.append(f"- {parts[0]} — {parts[1]}")
            if prev_lines:
                prev_text = "\n\nAlready recommended (DO NOT repeat these):\n" + "\n".join(prev_lines)

        user_prompt = (
            f"User wants: \"{prompt}\"\n\n"
            f"Clarifying answers:\n{answers_text}\n\n"
            f"User's taste profile:\n{taste_text}\n\n"
            f"Albums user already owns (DO NOT recommend these):\n{exclusion_text}"
            f"{prev_text}\n\n"
            f"Recommend 7 albums they don't own: 1 primary + 6 secondary."
        )

        response = self.llm_client.analyze(user_prompt, system)
        self._log_cost("discovery_selection", response, session_id)

        raw = self.llm_client.parse_json_response(response)
        recommendations = []

        # Build a set of owned albums for post-filtering (catches what the
        # prompt-level sample of 200 misses in large libraries)
        owned_set = {
            album_key(a["artist"], a["album"])
            for a in taste_profile.owned_albums
        }

        for item in raw[:7]:
            if len(recommendations) >= 3:
                break
            # Skip albums the user already owns
            rec_key = album_key(item.get("artist", ""), item.get("album", ""))
            if rec_key in owned_set:
                logger.info(
                    "Discovery post-filter: skipping owned album %s — %s",
                    item.get("artist"), item.get("album"),
                )
                continue

            rank = item.get("rank", "secondary")
            rec = AlbumRecommendation(
                rank=rank if rank in ("primary", "secondary") else "secondary",
                album=item.get("album", ""),
                artist=item.get("artist", ""),
                year=item.get("year"),
                rating_key=None,
                track_rating_keys=[],
                art_url=None,
            )
            recommendations.append(rec)

        if recommendations and all(r.rank == "secondary" for r in recommendations):
            recommendations[0].rank = "primary"

        return recommendations

    def validate_discovery_album(
        self,
        rec: AlbumRecommendation,
        research: ResearchData,
        prompt: str,
        session_id: str,
    ) -> bool:
        """Validate a discovery album against research data.

        Asks the LLM to confirm the album matches the user's request
        given the real research data. Returns True if valid.
        """
        research_text = f"Album: {rec.artist} — {rec.album}"
        if research.release_date:
            research_text += f"\nRelease date: {research.release_date}"
        if research.label:
            research_text += f"\nLabel: {research.label}"
        if research.genre_tags:
            research_text += f"\nGenres: {', '.join(research.genre_tags)}"
        if research.wikipedia_summary:
            research_text += f"\nAbout: {research.wikipedia_summary[:300]}"

        system = (
            "You are validating an album recommendation. Given the user's request and "
            "research data about the album, determine if this album genuinely matches "
            "the request in terms of genre, mood, and character.\n\n"
            "Return ONLY a JSON object: {\"valid\": true} or {\"valid\": false, \"reason\": \"...\"}"
        )

        user_prompt = (
            f"User wanted: \"{prompt}\"\n\n"
            f"Album research:\n{research_text}\n\n"
            f"Does this album genuinely match the request?"
        )

        response = self.llm_client.generate(user_prompt, system)
        self._log_cost("discovery_validation", response, session_id)

        result = self.llm_client.parse_json_response(response)
        if not isinstance(result, dict):
            logger.warning("validate_discovery_album: unexpected response type %s, treating as invalid", type(result).__name__)
            return False
        return result.get("valid", False)
