"""Tests for prompt and track analysis."""

import json
import pytest
from unittest.mock import MagicMock, patch


class TestPromptAnalysis:
    """Tests for prompt analysis."""

    @pytest.fixture(autouse=True)
    def no_cache(self, monkeypatch):
        monkeypatch.setattr("backend.analyzer.library_cache.has_cached_tracks", lambda: False)

    def _llm(self, reply):
        from backend.llm_client import LLMResponse

        client = MagicMock()
        client.analyze.return_value = LLMResponse(content=json.dumps(reply), input_tokens=1, output_tokens=1, model="m")
        client.parse_json_response.return_value = reply
        return client

    def test_works_with_jellyfin_client(self):
        """Prompt analysis used the Plex client directly, so it failed in Jellyfin mode."""
        from backend.analyzer import analyze_prompt
        from backend.jellyfin_client import JellyfinClient

        jellyfin = MagicMock(spec=JellyfinClient)
        jellyfin.get_library_stats.return_value = {"genres": [{"name": "Jazz", "count": 5}], "decades": []}
        with patch("backend.analyzer.get_llm_client", return_value=self._llm({"genres": ["Jazz"], "decades": []})), \
             patch("backend.analyzer.get_current_media_client", return_value=jellyfin):
            result = analyze_prompt("late night jazz")

        assert result.suggested_genres == ["Jazz"]

    def test_suggests_named_artists_in_library(self, monkeypatch):
        """'A Radiohead playlist' pre-fills the artist filter with library spellings (#18)."""
        from backend.analyzer import analyze_prompt

        monkeypatch.setattr("backend.analyzer.library_cache.has_cached_tracks", lambda: True)
        monkeypatch.setattr("backend.analyzer.library_cache.get_cached_genre_decade_stats",
                            lambda: {"genres": [], "decades": []})
        monkeypatch.setattr("backend.analyzer.library_cache.get_artist_names",
                            lambda: ["Radiohead", "The Beatles", "Sigur Rós"])
        reply = {"genres": [], "decades": [], "artists": ["radiohead", "Beatles", "Sigur Ros", "Taylor Swift"]}
        with patch("backend.analyzer.get_llm_client", return_value=self._llm(reply)), \
             patch("backend.analyzer.get_current_media_client", return_value=MagicMock()):
            result = analyze_prompt("Radiohead, the Beatles and Sigur Ros")

        assert result.suggested_artists == ["Radiohead", "The Beatles", "Sigur Rós"]

    def test_no_artist_suggestions_without_cache(self):
        from backend.analyzer import analyze_prompt

        media_client = MagicMock()
        media_client.get_library_stats.return_value = {"genres": [], "decades": []}
        with patch("backend.analyzer.get_llm_client", return_value=self._llm({"artists": ["Radiohead"]})), \
             patch("backend.analyzer.get_current_media_client", return_value=media_client):
            result = analyze_prompt("a Radiohead playlist")

        assert result.suggested_artists == []

    def test_uses_cached_genres_when_synced(self, monkeypatch):
        from backend.analyzer import analyze_prompt

        monkeypatch.setattr("backend.analyzer.library_cache.has_cached_tracks", lambda: True)
        monkeypatch.setattr(
            "backend.analyzer.library_cache.get_cached_genre_decade_stats",
            lambda: {"genres": [{"name": "Jazz", "count": 5}], "decades": [{"name": "1990s", "count": 5}]},
        )
        media_client = MagicMock()
        with patch("backend.analyzer.get_llm_client", return_value=self._llm({"genres": ["Jazz"], "decades": ["1990s"]})), \
             patch("backend.analyzer.get_current_media_client", return_value=media_client):
            result = analyze_prompt("90s jazz")

        assert (result.suggested_genres, result.suggested_decades) == (["Jazz"], ["1990s"])
        media_client.get_library_stats.assert_not_called()

    def test_analyze_prompt_extracts_genres(self, mocker):
        """Should extract suggested genres from prompt."""
        from backend.analyzer import analyze_prompt
        from backend.llm_client import LLMResponse

        mock_response = LLMResponse(
            content=json.dumps({
                "genres": ["Alternative", "Rock"],
                "decades": ["1990s"],
                "reasoning": "The prompt suggests 90s alternative rock."
            }),
            input_tokens=100,
            output_tokens=50,
            model="test-model"
        )

        with patch("backend.analyzer.get_llm_client") as mock_llm:
            mock_client = MagicMock()
            mock_client.analyze.return_value = mock_response
            mock_client.parse_json_response.return_value = json.loads(mock_response.content)
            mock_llm.return_value = mock_client

            with patch("backend.analyzer.get_current_media_client") as mock_plex:
                mock_plex_client = MagicMock()
                mock_plex_client.get_library_stats.return_value = {
                    "genres": [{"name": "Alternative", "count": 100}, {"name": "Rock", "count": 200}],
                    "decades": [{"name": "1990s", "count": 150}]
                }
                mock_plex.return_value = mock_plex_client

                result = analyze_prompt("melancholy 90s alternative")

                assert "Alternative" in result.suggested_genres
                assert "1990s" in result.suggested_decades

    def test_analyze_prompt_handles_malformed_response(self, mocker):
        """Should handle malformed LLM JSON responses gracefully."""
        from backend.analyzer import analyze_prompt
        from backend.llm_client import LLMResponse

        mock_response = LLMResponse(
            content="Not valid JSON at all",
            input_tokens=100,
            output_tokens=50,
            model="test-model"
        )

        with patch("backend.analyzer.get_llm_client") as mock_llm:
            mock_client = MagicMock()
            mock_client.analyze.return_value = mock_response
            mock_client.parse_json_response.side_effect = ValueError("Invalid JSON")
            mock_llm.return_value = mock_client

            with patch("backend.analyzer.get_current_media_client") as mock_plex:
                mock_plex_client = MagicMock()
                mock_plex_client.get_library_stats.return_value = {
                    "genres": [{"name": "Rock", "count": 100}],
                    "decades": [{"name": "1990s", "count": 100}]
                }
                mock_plex.return_value = mock_plex_client

                with pytest.raises(ValueError):
                    analyze_prompt("test prompt")

    def test_analyze_prompt_returns_available_filters(self, mocker):
        """Should return available genres and decades from library."""
        from backend.analyzer import analyze_prompt
        from backend.llm_client import LLMResponse

        mock_response = LLMResponse(
            content=json.dumps({
                "genres": ["Rock"],
                "decades": ["1990s"],
                "reasoning": "Test"
            }),
            input_tokens=100,
            output_tokens=50,
            model="test-model"
        )

        with patch("backend.analyzer.get_llm_client") as mock_llm:
            mock_client = MagicMock()
            mock_client.analyze.return_value = mock_response
            mock_client.parse_json_response.return_value = json.loads(mock_response.content)
            mock_llm.return_value = mock_client

            with patch("backend.analyzer.get_current_media_client") as mock_plex:
                mock_plex_client = MagicMock()
                mock_plex_client.get_library_stats.return_value = {
                    "genres": [
                        {"name": "Rock", "count": 500},
                        {"name": "Jazz", "count": 200},
                        {"name": "Classical", "count": 100}
                    ],
                    "decades": [
                        {"name": "1980s", "count": 300},
                        {"name": "1990s", "count": 400},
                        {"name": "2000s", "count": 200}
                    ]
                }
                mock_plex.return_value = mock_plex_client

                result = analyze_prompt("rock music from the 90s")

                # Should include all available options from library
                assert len(result.available_genres) == 3
                assert len(result.available_decades) == 3


class TestFilterSuggestions:
    """Tests for filter suggestion logic."""

    def test_filters_suggest_matching_library_genres(self, mocker):
        """Suggested genres should match library genres."""
        from backend.analyzer import analyze_prompt
        from backend.llm_client import LLMResponse

        # LLM suggests "Alt Rock" but library has "Alternative"
        mock_response = LLMResponse(
            content=json.dumps({
                "genres": ["Alt Rock", "Grunge"],
                "decades": ["1990s"],
                "reasoning": "90s alternative rock"
            }),
            input_tokens=100,
            output_tokens=50,
            model="test-model"
        )

        with patch("backend.analyzer.get_llm_client") as mock_llm:
            mock_client = MagicMock()
            mock_client.analyze.return_value = mock_response
            mock_client.parse_json_response.return_value = json.loads(mock_response.content)
            mock_llm.return_value = mock_client

            with patch("backend.analyzer.get_current_media_client") as mock_plex:
                mock_plex_client = MagicMock()
                mock_plex_client.get_library_stats.return_value = {
                    "genres": [
                        {"name": "Alternative", "count": 500},
                        {"name": "Grunge", "count": 100}
                    ],
                    "decades": [{"name": "1990s", "count": 400}]
                }
                mock_plex.return_value = mock_plex_client

                result = analyze_prompt("90s alt rock")

                # Grunge should be in suggestions (exact match)
                # Alt Rock might not be if no fuzzy match
                assert "Grunge" in result.suggested_genres


class TestTrackAnalysis:
    """Tests for seed track dimension extraction."""

    def test_analyze_track_extracts_dimensions(self, mocker):
        """Should extract musical dimensions from a track."""
        from backend.analyzer import analyze_track
        from backend.llm_client import LLMResponse
        from backend.models import Track

        track = Track(
            rating_key="1",
            title="Fake Plastic Trees",
            artist="Radiohead",
            album="The Bends",
            duration_ms=290000,
            year=1995,
            genres=["Alternative", "Rock"],
        )

        mock_response = LLMResponse(
            content=json.dumps({
                "dimensions": [
                    {"id": "mood", "label": "Melancholy, bittersweet mood", "description": "Emotional and reflective"},
                    {"id": "era", "label": "Mid-90s British alternative", "description": "Britpop era sound"},
                    {"id": "vocals", "label": "Falsetto-tinged vocals", "description": "Thom Yorke's distinctive style"},
                ]
            }),
            input_tokens=100,
            output_tokens=80,
            model="test-model"
        )

        with patch("backend.analyzer.get_llm_client") as mock_llm:
            mock_client = MagicMock()
            mock_client.analyze.return_value = mock_response
            mock_client.parse_json_response.return_value = json.loads(mock_response.content)
            mock_llm.return_value = mock_client

            result = analyze_track(track)

            assert result.track.rating_key == "1"
            assert len(result.dimensions) == 3
            assert result.dimensions[0].id == "mood"
            assert "melancholy" in result.dimensions[0].label.lower()

    def test_analyze_track_returns_specific_labels(self, mocker):
        """Dimension labels should be specific, not generic."""
        from backend.analyzer import analyze_track
        from backend.llm_client import LLMResponse
        from backend.models import Track

        track = Track(
            rating_key="2",
            title="Black",
            artist="Pearl Jam",
            album="Ten",
            duration_ms=340000,
            year=1991,
            genres=["Grunge", "Rock"],
        )

        mock_response = LLMResponse(
            content=json.dumps({
                "dimensions": [
                    {"id": "mood", "label": "The raw, aching heartbreak", "description": "Loss and longing"},
                    {"id": "vocals", "label": "Eddie Vedder's baritone intensity", "description": "Powerful delivery"},
                ]
            }),
            input_tokens=100,
            output_tokens=60,
            model="test-model"
        )

        with patch("backend.analyzer.get_llm_client") as mock_llm:
            mock_client = MagicMock()
            mock_client.analyze.return_value = mock_response
            mock_client.parse_json_response.return_value = json.loads(mock_response.content)
            mock_llm.return_value = mock_client

            result = analyze_track(track)

            # Labels should be specific, not just "the mood" or "the vocals"
            for dim in result.dimensions:
                assert len(dim.label) > 10  # Should be descriptive
                assert dim.label.lower() != f"the {dim.id}"  # Not generic
