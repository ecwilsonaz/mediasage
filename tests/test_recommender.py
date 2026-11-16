"""Tests for the recommendation pipeline grounding improvements."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.models import (
    ExtractedFacts,
    PitchIssue,
    PitchValidation,
    RecommendSessionState,
    ResearchData,
)
from backend.recommender import (
    MAX_SESSIONS,
    SESSION_EXPIRY,
    RecommendationPipeline,
)


class TestGroundingModels:
    """Test new Pydantic models for pitch grounding."""

    def test_extracted_facts_defaults(self):
        """ExtractedFacts should have sensible defaults for all fields."""
        facts = ExtractedFacts()
        assert facts.origin_story == ""
        assert facts.personnel == []
        assert facts.musical_style == ""
        assert facts.vocal_approach == ""
        assert facts.cultural_context == ""
        assert facts.track_highlights == ""
        assert facts.common_misconceptions == ""
        assert facts.source_coverage == ""

    def test_extracted_facts_populated(self):
        """ExtractedFacts should accept all fields."""
        facts = ExtractedFacts(
            origin_story="Recorded after Berman's death",
            personnel=["Cassandra Jenkins", "Stuart Bogie"],
            musical_style="Ambient folk",
            vocal_approach="Sung vocals with spoken word on Hard Drive",
            cultural_context="Released on Ba Da Bing! Records",
            track_highlights="Hard Drive features spoken word",
            common_misconceptions="Jenkins never toured with Berman",
            source_coverage="Wikipedia covers origin well; no reviews available",
        )
        assert "Berman" in facts.origin_story
        assert len(facts.personnel) == 2

    def test_pitch_validation_valid(self):
        """PitchValidation should represent a passing check."""
        result = PitchValidation(valid=True)
        assert result.valid is True
        assert result.issues == []

    def test_pitch_validation_with_issues(self):
        """PitchValidation should hold a list of issues."""
        result = PitchValidation(
            valid=False,
            issues=[
                PitchIssue(
                    claim="touring stint with David Berman",
                    problem="contradicts research",
                    correction="Jenkins rehearsed with Purple Mountains but Berman died before the tour began",
                ),
            ],
        )
        assert not result.valid
        assert len(result.issues) == 1
        assert "rehearsed" in result.issues[0].correction

    def test_research_data_review_texts(self):
        """ResearchData should support review_texts field."""
        rd = ResearchData(
            wikipedia_summary="Album summary here",
            review_texts=["Great album review from Pitchfork"],
        )
        assert len(rd.review_texts) == 1
        assert rd.review_texts[0].startswith("Great")

    def test_research_data_review_texts_default(self):
        """ResearchData.review_texts should default to empty list."""
        rd = ResearchData()
        assert rd.review_texts == []


class TestMusicResearchReviews:
    """Test review URL extraction and fetching in MusicResearchClient."""

    @pytest.mark.asyncio
    async def test_lookup_release_group_extracts_review_urls(self):
        """lookup_release_group should extract review-type URLs."""
        from backend.music_research import MusicResearchClient

        client = MusicResearchClient()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "relations": [
                {"type": "wikipedia", "url": {"resource": "https://en.wikipedia.org/wiki/Test_Album"}},
                {"type": "review", "url": {"resource": "https://pitchfork.com/reviews/albums/test/"}},
                {"type": "review", "url": {"resource": "https://stereogum.com/review/test/"}},
                {"type": "allmusic", "url": {"resource": "https://www.allmusic.com/album/test"}},
            ],
            "releases": [{"id": "rel-123", "date": "2020-01-01"}],
        }

        mock_http = AsyncMock()
        mock_http.is_closed = False
        mock_http.get = AsyncMock(return_value=mock_response)
        client._http = mock_http
        client._last_mb_request = 0

        result = await client.lookup_release_group("test-mbid")

        assert result is not None
        assert result["wikipedia_url"] == "https://en.wikipedia.org/wiki/Test_Album"
        assert "https://pitchfork.com/reviews/albums/test/" in result["review_urls"]
        assert "https://stereogum.com/review/test/" in result["review_urls"]
        # AllMusic should NOT be in review_urls (TOS)
        assert "https://www.allmusic.com/album/test" not in result.get("review_urls", [])

    @pytest.mark.asyncio
    async def test_fetch_review_text_extracts_article(self):
        """fetch_review_text should extract article text from HTML."""
        from backend.music_research import MusicResearchClient

        client = MusicResearchClient()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_redirect = False
        mock_response.raise_for_status = MagicMock()
        mock_response.text = """
        <html><head><title>Album Review</title></head>
        <body>
        <nav>Site Navigation</nav>
        <article>
        <h1>Album Review: Test Album</h1>
        <p>This is a detailed review of the album. The recording was made in a
        studio in Brooklyn with producer John Smith. The vocals are predominantly
        sung in English with some instrumental passages.</p>
        </article>
        <footer>Site Footer</footer>
        </body></html>
        """

        mock_http = AsyncMock()
        mock_http.is_closed = False
        mock_http.get = AsyncMock(return_value=mock_response)
        client._http = mock_http

        result = await client.fetch_review_text("https://example.com/review")

        assert result is not None
        assert "detailed review" in result
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_fetch_review_text_skips_allmusic(self):
        """fetch_review_text should refuse to fetch AllMusic URLs."""
        from backend.music_research import MusicResearchClient

        client = MusicResearchClient()
        result = await client.fetch_review_text("https://www.allmusic.com/album/test")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_review_text_handles_failure(self):
        """fetch_review_text should return None on HTTP errors."""
        from backend.music_research import MusicResearchClient

        client = MusicResearchClient()
        mock_http = AsyncMock()
        mock_http.is_closed = False
        mock_http.get = AsyncMock(side_effect=Exception("Connection refused"))
        client._http = mock_http

        result = await client.fetch_review_text("https://example.com/review")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_review_text_truncates_long_content(self):
        """fetch_review_text should truncate to ~2000 chars."""
        from backend.music_research import MusicResearchClient

        client = MusicResearchClient()
        long_text = "<html><body><article>" + ("x" * 5000) + "</article></body></html>"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_redirect = False
        mock_response.raise_for_status = MagicMock()
        mock_response.text = long_text

        mock_http = AsyncMock()
        mock_http.is_closed = False
        mock_http.get = AsyncMock(return_value=mock_response)
        client._http = mock_http

        result = await client.fetch_review_text("https://example.com/review")

        assert result is not None
        assert len(result) <= 2100  # Allow small margin for sentence boundary


class TestFactExtraction:
    """Test the LLM fact extraction step."""

    def test_extract_facts_returns_structured_data(self):
        """extract_facts should return an ExtractedFacts from research data."""
        from backend.recommender import RecommendationPipeline

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"origin_story": "Recorded in Reykjavik"}'
        mock_response.input_tokens = 100
        mock_response.output_tokens = 50
        mock_response.model = "test-model"
        mock_response.estimated_cost.return_value = 0.001
        mock_llm.generate.return_value = mock_response
        mock_llm.parse_json_response.return_value = {
            "origin_story": "Recorded in Reykjavik",
            "personnel": ["Jonsi"],
            "musical_style": "Post-rock with orchestral arrangements",
            "vocal_approach": "Mostly Icelandic vocals, Vonlenska on 2 tracks only",
            "cultural_context": "Breakthrough album internationally",
            "track_highlights": "Svefn-g-englar is the lead single",
            "common_misconceptions": "Often assumed to be entirely in Vonlenska but most tracks are in Icelandic",
            "source_coverage": "Wikipedia covers recording and reception well",
        }

        pipeline = RecommendationPipeline(config=MagicMock(), llm_client=mock_llm)

        rd = ResearchData(
            wikipedia_summary="Agaetis byrjun is the second album by Sigur Ros...",
            track_listing=["Intro", "Svefn-g-englar", "Staralfur"],
            label="Smekkleysa",
            release_date="1999-06-12",
            credits={"Primary Artist": "Sigur Ros"},
            review_texts=["A landmark post-rock album..."],
        )

        facts = pipeline.extract_facts(
            artist="Sigur Ros",
            album="Agaetis byrjun",
            research=rd,
            session_id="test-session",
        )

        assert isinstance(facts, ExtractedFacts)
        assert "Vonlenska" in facts.vocal_approach
        assert "Icelandic" in facts.vocal_approach
        mock_llm.generate.assert_called_once()

    def test_extract_facts_includes_all_sources_in_prompt(self):
        """extract_facts should pass Wikipedia, reviews, and track listing to LLM."""
        from backend.recommender import RecommendationPipeline

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.input_tokens = 100
        mock_response.output_tokens = 50
        mock_response.model = "test-model"
        mock_response.estimated_cost.return_value = 0.001
        mock_llm.generate.return_value = mock_response
        mock_llm.parse_json_response.return_value = {
            "origin_story": "", "personnel": [], "musical_style": "",
            "vocal_approach": "", "cultural_context": "", "track_highlights": "",
            "common_misconceptions": "", "source_coverage": "",
        }

        pipeline = RecommendationPipeline(config=MagicMock(), llm_client=mock_llm)

        rd = ResearchData(
            wikipedia_summary="Wikipedia content here",
            review_texts=["Pitchfork review content", "Stereogum review content"],
            track_listing=["Track One", "Track Two"],
            label="Test Label",
            release_date="2020",
        )

        pipeline.extract_facts(
            artist="Test Artist", album="Test Album",
            research=rd, session_id="test",
        )

        # Verify the prompt includes all source material
        call_args = mock_llm.generate.call_args
        prompt = call_args[0][0]  # First positional arg is the user prompt
        assert "Wikipedia content here" in prompt
        assert "Pitchfork review content" in prompt
        assert "Stereogum review content" in prompt
        assert "Track One" in prompt
        assert "Test Label" in prompt

    def test_extract_facts_handles_empty_research(self):
        """extract_facts should handle research with no Wikipedia or reviews."""
        from backend.recommender import RecommendationPipeline

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.input_tokens = 10
        mock_response.output_tokens = 10
        mock_response.model = "test-model"
        mock_response.estimated_cost.return_value = 0.0
        mock_llm.generate.return_value = mock_response
        mock_llm.parse_json_response.return_value = {
            "origin_story": "NOT IN SOURCES",
            "personnel": [],
            "musical_style": "NOT IN SOURCES",
            "vocal_approach": "NOT IN SOURCES",
            "cultural_context": "NOT IN SOURCES",
            "track_highlights": "",
            "common_misconceptions": "",
            "source_coverage": "No Wikipedia or review sources available",
        }

        pipeline = RecommendationPipeline(config=MagicMock(), llm_client=mock_llm)

        rd = ResearchData(
            label="Test Label",
            track_listing=["Track 1"],
        )

        facts = pipeline.extract_facts(
            artist="Test", album="Test", research=rd, session_id="test",
        )

        assert isinstance(facts, ExtractedFacts)


class TestWritePitchesGrounding:
    """Test that write_pitches uses structured facts and grounding rules."""

    def test_write_pitches_uses_extracted_facts(self):
        """write_pitches should include ExtractedFacts in the prompt, not raw Wikipedia."""
        from backend.recommender import RecommendationPipeline
        from backend.models import (
            AlbumRecommendation,
            ExtractedFacts,
            ResearchData,
        )

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.input_tokens = 200
        mock_response.output_tokens = 300
        mock_response.model = "test-model"
        mock_response.estimated_cost.return_value = 0.01
        mock_llm.analyze.return_value = mock_response
        mock_llm.parse_json_response.return_value = [
            {
                "artist": "Sigur Ros",
                "album": "Agaetis byrjun",
                "hook": "A test hook",
                "context": "A test context",
                "listening_guide": "A test guide",
                "connection": "A test connection",
            },
        ]

        pipeline = RecommendationPipeline(config=MagicMock(), llm_client=mock_llm)

        recs = [
            AlbumRecommendation(
                rank="primary",
                album="Agaetis byrjun",
                artist="Sigur Ros",
                year=1999,
                rating_key="123",
                track_rating_keys=["456"],
            ),
        ]

        facts = {
            "sigur ros|||agaetis byrjun": ExtractedFacts(
                origin_story="Recorded in a Reykjavik swimming pool",
                vocal_approach="Mostly Icelandic; Vonlenska on 2 tracks only",
                common_misconceptions="Not entirely in Vonlenska despite common belief",
            ),
        }

        research = {
            "sigur ros|||agaetis byrjun": ResearchData(
                track_listing=["Intro", "Svefn-g-englar", "Staralfur"],
            ),
        }

        pipeline.write_pitches(
            recommendations=recs,
            prompt="something atmospheric",
            answers=[None],
            answer_texts=[],
            session_id="test",
            research=research,
            extracted_facts=facts,
        )

        call_args = mock_llm.analyze.call_args
        system_prompt = call_args[0][1]
        user_prompt = call_args[0][0]

        # Grounding rules should be in system prompt
        assert "GROUNDING RULES" in system_prompt
        assert "NOT IN SOURCES" in system_prompt

        # Extracted facts should be in user prompt, not raw Wikipedia
        assert "Vonlenska on 2 tracks only" in user_prompt
        assert "common misconceptions" in user_prompt.lower()

        # Track listing should be included
        assert "Svefn-g-englar" in user_prompt

    def test_write_pitches_no_500_char_truncation(self):
        """write_pitches should NOT truncate Wikipedia to 500 chars (old behavior)."""
        from backend.recommender import RecommendationPipeline
        from backend.models import AlbumRecommendation

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.input_tokens = 200
        mock_response.output_tokens = 300
        mock_response.model = "test-model"
        mock_response.estimated_cost.return_value = 0.01
        mock_llm.analyze.return_value = mock_response
        mock_llm.parse_json_response.return_value = [
            {"artist": "Test", "album": "Test", "hook": "", "context": "",
             "listening_guide": "", "connection": ""},
        ]

        pipeline = RecommendationPipeline(config=MagicMock(), llm_client=mock_llm)

        recs = [AlbumRecommendation(
            rank="primary", album="Test", artist="Test",
            rating_key="1", track_rating_keys=["2"],
        )]

        # No extracted_facts passed — should still work without [:500] truncation
        pipeline.write_pitches(
            recommendations=recs, prompt="test", answers=[],
            answer_texts=[], session_id="test",
        )

        # Should not crash and should produce a pitch
        assert recs[0].pitch.hook is not None


class TestPitchValidation:
    """Test pitch validation against research data."""

    def test_validate_pitch_passes_clean_pitch(self):
        """validate_pitch should return valid=True for a factually correct pitch."""
        from backend.recommender import RecommendationPipeline
        from backend.models import SommelierPitch

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.input_tokens = 100
        mock_response.output_tokens = 20
        mock_response.model = "test-model"
        mock_response.estimated_cost.return_value = 0.005
        mock_llm.analyze.return_value = mock_response
        mock_llm.parse_json_response.return_value = {"valid": True, "issues": []}

        pipeline = RecommendationPipeline(config=MagicMock(), llm_client=mock_llm)

        pitch = SommelierPitch(
            hook="A good hook",
            context="Recorded in Reykjavik",
            listening_guide="Listen for the strings",
            connection="Matches your request",
            full_text="A good hook\n\nRecorded in Reykjavik",
        )
        facts = ExtractedFacts(
            origin_story="Recorded in Reykjavik",
            musical_style="Post-rock with orchestral elements",
        )

        result = pipeline.validate_pitch(
            pitch=pitch, facts=facts, session_id="test",
        )

        assert result.valid is True
        assert len(result.issues) == 0

    def test_validate_pitch_catches_inaccuracy(self):
        """validate_pitch should flag claims contradicting research."""
        from backend.recommender import RecommendationPipeline
        from backend.models import SommelierPitch

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.input_tokens = 100
        mock_response.output_tokens = 100
        mock_response.model = "test-model"
        mock_response.estimated_cost.return_value = 0.005
        mock_llm.analyze.return_value = mock_response
        mock_llm.parse_json_response.return_value = {
            "valid": False,
            "issues": [
                {
                    "claim": "touring stint with David Berman",
                    "problem": "contradicts research",
                    "correction": "Jenkins rehearsed with Purple Mountains but Berman died before the tour began",
                }
            ],
        }

        pipeline = RecommendationPipeline(config=MagicMock(), llm_client=mock_llm)

        pitch = SommelierPitch(full_text="Born from her touring stint with David Berman...")
        facts = ExtractedFacts(
            origin_story="Jenkins rehearsed with Purple Mountains for four days before Berman died",
        )

        result = pipeline.validate_pitch(
            pitch=pitch, facts=facts, session_id="test",
        )

        assert not result.valid
        assert len(result.issues) == 1
        assert "rehearsed" in result.issues[0].correction


class TestPitchRewrite:
    """Test pitch rewriting with corrections."""

    def test_rewrite_pitch_incorporates_corrections(self):
        """rewrite_pitch should pass corrections as constraints to the LLM."""
        from backend.recommender import RecommendationPipeline
        from backend.models import (
            AlbumRecommendation,
            PitchIssue,
            PitchValidation,
        )

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.input_tokens = 200
        mock_response.output_tokens = 300
        mock_response.model = "test-model"
        mock_response.estimated_cost.return_value = 0.01
        mock_llm.analyze.return_value = mock_response
        mock_llm.parse_json_response.return_value = {
            "hook": "Corrected hook",
            "context": "Jenkins rehearsed with the band before tragedy struck",
            "listening_guide": "Guide",
            "connection": "Connection",
        }

        pipeline = RecommendationPipeline(config=MagicMock(), llm_client=mock_llm)

        rec = AlbumRecommendation(
            rank="primary",
            album="An Overview on Phenomenal Nature",
            artist="Cassandra Jenkins",
            year=2021,
        )
        facts = ExtractedFacts(
            origin_story="Jenkins rehearsed with Purple Mountains for four days",
        )
        validation = PitchValidation(
            valid=False,
            issues=[
                PitchIssue(
                    claim="touring stint with David Berman",
                    problem="contradicts research",
                    correction="Rehearsed for four days, never toured",
                ),
            ],
        )

        pipeline.rewrite_pitch(
            rec=rec,
            facts=facts,
            validation=validation,
            prompt="something contemplative",
            answers_str="calm; introspective",
            session_id="test",
        )

        # Verify corrections were in the prompt
        call_args = mock_llm.analyze.call_args
        prompt = call_args[0][0]
        assert "touring stint with David Berman" in prompt
        assert "Rehearsed for four days" in prompt

        # Verify pitch was updated
        assert rec.pitch.hook == "Corrected hook"


# ── Session Management Tests ────────────────────────────────────────────


def _make_pipeline():
    """Create a pipeline with mock dependencies."""
    config = MagicMock()
    config.llm = MagicMock()
    llm_client = MagicMock()
    return RecommendationPipeline(config, llm_client)


class TestSessionLifecycle:
    """Test session create/get/update/delete and expiry."""

    def test_create_and_get(self):
        pipeline = _make_pipeline()
        state = RecommendSessionState(prompt="jazz recs")
        sid = pipeline.create_session(state)

        assert sid.startswith("rec_")
        retrieved = pipeline.get_session(sid)
        assert retrieved is not None
        assert retrieved.prompt == "jazz recs"

    def test_get_nonexistent_returns_none(self):
        pipeline = _make_pipeline()
        assert pipeline.get_session("rec_doesnotexist") is None

    def test_delete_session(self):
        pipeline = _make_pipeline()
        state = RecommendSessionState(prompt="test")
        sid = pipeline.create_session(state)
        pipeline.delete_session(sid)
        assert pipeline.get_session(sid) is None

    def test_update_answers(self):
        pipeline = _make_pipeline()
        state = RecommendSessionState(prompt="test")
        sid = pipeline.create_session(state)

        pipeline.update_session_answers(sid, ["option_a", None], ["Rock", ""])
        retrieved = pipeline.get_session(sid)
        assert retrieved.answers == ["option_a", None]
        assert retrieved.answer_texts == ["Rock", ""]

    def test_update_previously_recommended(self):
        pipeline = _make_pipeline()
        state = RecommendSessionState(prompt="test")
        sid = pipeline.create_session(state)

        pipeline.update_previously_recommended(sid, ["artist1|||album1"])
        retrieved = pipeline.get_session(sid)
        assert "artist1|||album1" in retrieved.previously_recommended

        # Duplicates should not be added
        pipeline.update_previously_recommended(sid, ["artist1|||album1"])
        retrieved = pipeline.get_session(sid)
        assert retrieved.previously_recommended.count("artist1|||album1") == 1

    def test_previously_recommended_capped_at_30(self):
        pipeline = _make_pipeline()
        state = RecommendSessionState(prompt="test")
        sid = pipeline.create_session(state)

        keys = [f"artist{i}|||album{i}" for i in range(35)]
        pipeline.update_previously_recommended(sid, keys)
        retrieved = pipeline.get_session(sid)
        assert len(retrieved.previously_recommended) == 30
        # Should keep the most recent 30
        assert retrieved.previously_recommended[-1] == "artist34|||album34"

    def test_session_expiry(self):
        pipeline = _make_pipeline()
        state = RecommendSessionState(prompt="old session")
        sid = pipeline.create_session(state)

        # Manually backdate the timestamp beyond SESSION_EXPIRY
        with pipeline._session_lock:
            s, _ = pipeline._sessions[sid]
            pipeline._sessions[sid] = (s, time.time() - SESSION_EXPIRY - 1)

        # Accessing triggers expiry check
        assert pipeline.get_session(sid) is None

    def test_session_eviction_over_cap(self):
        pipeline = _make_pipeline()

        # Create MAX_SESSIONS + 5 sessions
        sids = []
        for i in range(MAX_SESSIONS + 5):
            state = RecommendSessionState(prompt=f"session {i}")
            sid = pipeline.create_session(state)
            sids.append(sid)

        # Eviction runs before each insert, so after the last insert there can
        # be MAX_SESSIONS + 1. Next access triggers cleanup.
        pipeline.get_session(sids[-1])
        with pipeline._session_lock:
            assert len(pipeline._sessions) <= MAX_SESSIONS

    def test_migrate_sessions(self):
        old = _make_pipeline()
        state = RecommendSessionState(prompt="migrated")
        sid = old.create_session(state)

        new = _make_pipeline()
        new.migrate_sessions_from(old)

        retrieved = new.get_session(sid)
        assert retrieved is not None
        assert retrieved.prompt == "migrated"

    def test_generate_state_resets_cost(self):
        pipeline = _make_pipeline()
        state = RecommendSessionState(prompt="test", total_tokens=500, total_cost=0.05)
        sid = pipeline.create_session(state)

        pipeline.update_session_generate_state(
            sid, mode="library", filters={}, familiarity_pref="any"
        )

        tokens, cost = pipeline.get_session_costs(sid)
        assert tokens == 0
        assert cost == 0.0


class TestPromptFilterCost:
    def test_filter_analysis_reports_its_cost(self):
        from backend.llm_client import LLMResponse
        from backend.recommender import RecommendationPipeline

        response = LLMResponse(content="{}", input_tokens=800, output_tokens=200, model="gemini-3.5-flash-lite")
        llm = MagicMock()
        llm.generate.return_value = response
        llm.parse_json_response.return_value = {"genres": ["Jazz"], "decades": [], "reasoning": "r"}
        result = RecommendationPipeline(config=MagicMock(), llm_client=llm).analyze_prompt_filters(
            "jazz", ["Jazz", "Rock"], ["1990s"]
        )
        assert result["genres"] == ["Jazz"]
        assert result["token_count"] == 1000
        assert result["estimated_cost"] == pytest.approx(response.estimated_cost())
