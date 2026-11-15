"""MusicBrainz and Wikipedia API integration for research-grounded pitches.

Fetches external data to ground sommelier pitches in verifiable facts:
release dates, personnel, recording context, and critical reception.
"""

import asyncio
import ipaddress
import logging
import re
import socket
import time
from urllib.parse import unquote, urlparse

import httpx
from readability import Document as ReadableDocument

from backend.models import ResearchData

logger = logging.getLogger(__name__)

# MusicBrainz requires a User-Agent header
USER_AGENT = "MediaSage/1.0 (https://github.com/ecwilsonaz/mediasage)"

# Rate limiting: 1 request/second to MusicBrainz
MB_BASE_URL = "https://musicbrainz.org/ws/2"
MB_RATE_LIMIT = 1.0  # seconds between requests

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

# Release-group scoring weights for _pick_best_release_group
RG_ARTIST_MATCH = 60       # Artist credit matches query artist
RG_TITLE_EXACT = 50        # Album title is an exact match
RG_TITLE_PREFIX = 30       # Album title starts with the query
RG_TITLE_CONTAINS = 10     # Query appears somewhere in the title
RG_TYPE_ALBUM = 20         # Primary type is "Album" (vs compilation/EP)
RG_YEAR_MATCH = 40         # First-release year matches query year
RG_MB_SCORE_DIVISOR = 10   # Normalise MusicBrainz relevance (0-100 → 0-10)

# Keywords to drop Wikipedia sections — matched as substrings against section titles
_WIKIPEDIA_DROP_KEYWORDS = [
    "track listing", "chart", "certification", "personnel", "credits",
    "reference", "external link", "see also", "note", "footnote",
    "accolade", "award", "release history", "singles", "bibliography",
    "further reading", "citation", "reissue", "remaster",
]
_WIKIPEDIA_MAX_CHARS = 8000  # Safety cap after filtering
COVER_ART_BASE = "https://coverartarchive.org"


def _filter_wikipedia_sections(text: str) -> str:
    """Filter Wikipedia plain text to keep only useful sections.

    Splits on MediaWiki section headers (== Title ==), drops sections
    whose titles contain any keyword from _WIKIPEDIA_DROP_KEYWORDS,
    and rejoins the rest. The lead section (before any header) is always
    kept. Result is capped at _WIKIPEDIA_MAX_CHARS on a paragraph boundary.
    """
    # Split on == Section == headers, keeping the headers
    parts = re.split(r"(^={2,}\s*.+?\s*={2,}\s*$)", text, flags=re.MULTILINE)

    result = []
    skip = False
    for part in parts:
        # Check if this part is a section header
        header_match = re.match(r"^={2,}\s*(.+?)\s*={2,}\s*$", part.strip())
        if header_match:
            section_name = header_match.group(1).strip().lower()
            skip = any(kw in section_name for kw in _WIKIPEDIA_DROP_KEYWORDS)
            if not skip:
                result.append(part)
        elif not skip:
            result.append(part)

    filtered = "".join(result).strip()

    # Cap on a paragraph boundary to avoid mid-sentence cuts
    if len(filtered) > _WIKIPEDIA_MAX_CHARS:
        cut = filtered[:_WIKIPEDIA_MAX_CHARS].rfind("\n\n")
        if cut > _WIKIPEDIA_MAX_CHARS // 2:
            filtered = filtered[:cut].strip()
        else:
            filtered = filtered[:_WIKIPEDIA_MAX_CHARS].strip()

    return filtered


def _is_safe_url(url: str) -> bool:
    """Validate that a URL is safe to fetch (not a private/internal address).

    NOTE: Subject to DNS rebinding (TOCTOU between resolution and fetch).
    Acceptable here because URLs come from MusicBrainz relations, not user input.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        for info in socket.getaddrinfo(hostname, None, socket.AF_UNSPEC):
            addr = info[4][0]
            ip = ipaddress.ip_address(addr)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        return True
    except (ValueError, socket.gaierror):
        return False


class MusicResearchClient:
    """Client for fetching album research data from MusicBrainz and Wikipedia."""

    def __init__(self):
        self._http: httpx.AsyncClient | None = None
        self._last_mb_request: float = 0
        self._client_lock = asyncio.Lock()  # Guards HTTP client init
        self._rate_lock = asyncio.Lock()  # Guards MusicBrainz rate limiting

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily create the HTTP client."""
        if self._http is None or self._http.is_closed:
            async with self._client_lock:
                if self._http is None or self._http.is_closed:
                    self._http = httpx.AsyncClient(
                        timeout=10.0,
                        headers={"User-Agent": USER_AGENT},
                    )
        return self._http

    async def _rate_limit(self) -> None:
        """Enforce MusicBrainz rate limiting (1 req/sec)."""
        async with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_mb_request
            if elapsed < MB_RATE_LIMIT:
                await asyncio.sleep(MB_RATE_LIMIT - elapsed)
            self._last_mb_request = time.monotonic()

    @staticmethod
    def _clean_album_name(album: str) -> str | None:
        """Strip common Plex parenthetical suffixes from album names.

        Returns the cleaned name, or None if nothing was stripped.
        """
        cleaned = re.sub(
            r"\s*\("
            r"(?:Explicit|Clean|Deluxe|Special|Expanded|Anniversary|Limited|"
            r"Bonus Track|Collector(?:'s)?|International|Standard|Super Deluxe|"
            r"Premium|Platinum|Ultimate|Complete|Original|Extended)"
            r"[^)]*\)\s*$",
            "",
            album,
            flags=re.IGNORECASE,
        ).strip()
        return cleaned if cleaned and cleaned != album else None

    async def search_album(
        self, artist: str, album: str, year: int | None = None
    ) -> str | None:
        """Search MusicBrainz for a release group by artist+album.

        Tries three strategies in order:
        1. Strict: artist + full album name
        2. Cleaned: artist + album name without parenthetical suffixes
           (handles Plex metadata like "Explicit Version", "Deluxe Edition")
        3. Album-only fallback with scoring
           (handles mismatched artist names: soundtracks, cast recordings)

        Returns the release group MBID, or None if not found.
        """
        client = await self._get_client()
        await self._rate_limit()

        # Step 1: Strict search — artist + full album name
        query = f'artist:"{artist}" AND releasegroup:"{album}"'
        try:
            resp = await client.get(
                f"{MB_BASE_URL}/release-group",
                params={"query": query, "fmt": "json", "limit": 5},
            )
            resp.raise_for_status()
            data = resp.json()

            release_groups = data.get("release-groups", [])
            if release_groups:
                return release_groups[0].get("id")
        except Exception as e:
            logger.warning("MusicBrainz strict search failed for %s — %s: %s", artist, album, e)
            # Fall through to next strategy rather than aborting

        # Step 2: Cleaned search — strip Plex parenthetical suffixes
        cleaned = self._clean_album_name(album)
        if cleaned:
            logger.info("Trying cleaned album name: %s → %s", album, cleaned)
            await self._rate_limit()

            query = f'artist:"{artist}" AND releasegroup:"{cleaned}"'
            try:
                resp = await client.get(
                    f"{MB_BASE_URL}/release-group",
                    params={"query": query, "fmt": "json", "limit": 5},
                )
                resp.raise_for_status()
                data = resp.json()

                release_groups = data.get("release-groups", [])
                if release_groups:
                    return release_groups[0].get("id")
            except Exception as e:
                logger.warning("MusicBrainz cleaned search failed: %s", e)

        # Step 3: Album-only fallback (handles mismatched artist names)
        search_name = cleaned or album
        logger.info("Strict search missed for %s — %s, trying album-only fallback", artist, album)
        await self._rate_limit()

        query = f'releasegroup:"{search_name}"'
        try:
            resp = await client.get(
                f"{MB_BASE_URL}/release-group",
                params={"query": query, "fmt": "json", "limit": 10},
            )
            resp.raise_for_status()
            data = resp.json()

            candidates = data.get("release-groups", [])
            if not candidates:
                logger.info("No MusicBrainz match for %s — %s (fallback)", artist, album)
                return None

            return self._pick_best_release_group(candidates, search_name, year, artist)
        except Exception as e:
            logger.warning("MusicBrainz fallback search failed for %s — %s: %s", artist, album, e)
            return None

    @staticmethod
    def _pick_best_release_group(
        candidates: list[dict], album: str, year: int | None,
        original_artist: str | None = None,
    ) -> str | None:
        """Score and pick the best release group from fallback results.

        Prefers: artist match, exact title match, Album type, year match, higher MB score.
        """
        album_lower = album.lower()
        artist_lower = original_artist.lower() if original_artist else None
        best_id = None
        best_score = -1

        for rg in candidates:
            score = 0
            title = rg.get("title", "")
            title_lower = title.lower()

            # Artist match (strong signal to avoid wrong-artist albums)
            if artist_lower:
                for credit in rg.get("artist-credit", []):
                    credit_name = credit.get("name", "").lower()
                    if artist_lower == credit_name or (
                        len(credit_name) >= 3
                        and (artist_lower in credit_name or credit_name in artist_lower)
                    ):
                        score += RG_ARTIST_MATCH
                        break

            # Title match: exact > starts-with > contains
            if title_lower == album_lower:
                score += RG_TITLE_EXACT
            elif title_lower.startswith(album_lower):
                score += RG_TITLE_PREFIX
            elif album_lower in title_lower:
                score += RG_TITLE_CONTAINS

            # Prefer Album type over Other/unknown
            if rg.get("primary-type") == "Album":
                score += RG_TYPE_ALBUM

            # Year match is a strong signal
            if year:
                release_date = rg.get("first-release-date", "")
                if release_date.startswith(str(year)):
                    score += RG_YEAR_MATCH

            # MB relevance score (0-100, normalized)
            mb_score = rg.get("score", 0)
            score += mb_score / RG_MB_SCORE_DIVISOR

            if score > best_score:
                best_score = score
                best_id = rg.get("id")

        if best_id:
            logger.info("Fallback picked release group %s (score=%d)", best_id, best_score)
        return best_id

    async def lookup_release_group(self, mbid: str) -> dict | None:
        """Look up a release group by MBID with URL rels and releases.

        Returns dict with wikipedia_url, allmusic_url, discogs_url,
        earliest_release_mbid, or None on failure.
        """
        client = await self._get_client()
        await self._rate_limit()

        try:
            resp = await client.get(
                f"{MB_BASE_URL}/release-group/{mbid}",
                params={"inc": "url-rels+releases", "fmt": "json"},
            )
            resp.raise_for_status()
            data = resp.json()

            result: dict = {}
            review_urls = []

            # Extract URLs from relations
            for rel in data.get("relations", []):
                rel_type = rel.get("type", "")
                url = rel.get("url", {}).get("resource", "")
                if rel_type == "wikipedia":
                    result["wikipedia_url"] = url
                elif rel_type == "wikidata":
                    result["wikidata_url"] = url
                elif rel_type == "discogs":
                    result["discogs_url"] = url
                elif rel_type == "review":
                    # Skip AllMusic (TOS prohibits automated access)
                    if "allmusic.com" not in url:
                        review_urls.append(url)

            result["review_urls"] = review_urls[:2]  # Limit to 2 reviews

            # Find earliest release MBID
            releases = data.get("releases", [])
            if releases:
                # Sort by date (earliest first)
                releases.sort(key=lambda r: r.get("date", "9999"))
                result["earliest_release_mbid"] = releases[0].get("id")
                result["release_date"] = releases[0].get("date")

            return result
        except Exception as e:
            logger.warning("MusicBrainz release group lookup failed for %s: %s", mbid, e)
            return None

    async def lookup_release(self, release_mbid: str) -> dict | None:
        """Look up a release by MBID for track listing, label, and credits.

        Returns dict with track_listing, label, credits, or None on failure.
        """
        client = await self._get_client()
        await self._rate_limit()

        try:
            resp = await client.get(
                f"{MB_BASE_URL}/release/{release_mbid}",
                params={"inc": "recordings+labels+artist-credits", "fmt": "json"},
            )
            resp.raise_for_status()
            data = resp.json()

            result: dict = {}

            # Track listing
            tracks = []
            for medium in data.get("media", []):
                for track in medium.get("tracks", []):
                    title = track.get("title", "")
                    if title:
                        tracks.append(title)
            result["track_listing"] = tracks

            # Label
            label_info = data.get("label-info", [])
            if label_info:
                label = label_info[0].get("label", {})
                result["label"] = label.get("name")

            # Credits from artist-credit
            artist_credit = data.get("artist-credit", [])
            credits = {}
            for credit in artist_credit:
                artist = credit.get("artist", {})
                if artist.get("name"):
                    credits["Primary Artist"] = artist["name"]
                    break
            result["credits"] = credits

            return result
        except Exception as e:
            logger.warning("MusicBrainz release lookup failed for %s: %s", release_mbid, e)
            return None

    async def fetch_wikipedia_summary(self, wikipedia_url: str) -> str | None:
        """Fetch article extract from Wikipedia, keeping only useful sections.

        Uses the MediaWiki extracts API to get the full article as plain text,
        then drops low-value sections (charts, track listing, references, etc.)
        to keep only content useful for fact-grounding pitches.

        Args:
            wikipedia_url: Full Wikipedia article URL

        Returns:
            Filtered article text, or None on failure.
        """
        client = await self._get_client()

        try:
            # Extract article title from URL
            # e.g. https://en.wikipedia.org/wiki/Spirit_of_Eden -> Spirit_of_Eden
            parts = wikipedia_url.rstrip("/").split("/wiki/")
            if len(parts) < 2:
                return None
            title = unquote(parts[1])

            resp = await client.get(WIKIPEDIA_API, params={
                "action": "query",
                "titles": title,
                "prop": "extracts",
                "explaintext": "true",
                "format": "json",
            })
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return None
            page = next(iter(pages.values()))
            full_text = page.get("extract", "")
            if not full_text:
                return None

            return _filter_wikipedia_sections(full_text)
        except Exception as e:
            logger.warning("Wikipedia fetch failed for %s: %s", wikipedia_url, e)
            return None

    async def resolve_wikidata_to_wikipedia(self, wikidata_url: str) -> str | None:
        """Resolve a Wikidata URL to an English Wikipedia article URL.

        Args:
            wikidata_url: Full Wikidata URL (e.g. https://www.wikidata.org/wiki/Q202996)

        Returns:
            Wikipedia URL, or None if no English Wikipedia article exists.
        """
        client = await self._get_client()

        try:
            qid = wikidata_url.rstrip("/").split("/")[-1]
            if not qid.startswith("Q"):
                return None

            resp = await client.get(
                f"https://www.wikidata.org/w/rest.php/wikibase/v1/entities/items/{qid}/sitelinks/enwiki"
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("url")
            return None
        except Exception as e:
            logger.warning("Wikidata resolution failed for %s: %s", wikidata_url, e)
            return None

    async def fetch_cover_art(
        self, release_mbid: str, release_group_mbid: str | None = None,
    ) -> str | None:
        """Fetch front cover art URL from Cover Art Archive.

        Tries the specific release first, then falls back to the release-group
        endpoint which aggregates art from all releases in the group.

        Returns the final image URL after redirect, or None if unavailable.
        """
        client = await self._get_client()

        # Try specific release first
        try:
            resp = await client.get(
                f"{COVER_ART_BASE}/release/{release_mbid}/front",
                follow_redirects=True,
            )
            if resp.status_code == 200:
                return str(resp.url)
        except Exception as e:
            logger.warning("Cover Art Archive fetch failed for release %s: %s", release_mbid, e)

        # Fall back to release-group (aggregates art from all editions)
        if release_group_mbid:
            try:
                resp = await client.get(
                    f"{COVER_ART_BASE}/release-group/{release_group_mbid}/front",
                    follow_redirects=True,
                )
                if resp.status_code == 200:
                    return str(resp.url)
            except Exception as e:
                logger.warning(
                    "Cover Art Archive fetch failed for release-group %s: %s",
                    release_group_mbid, e,
                )

        return None

    async def fetch_review_text(self, url: str) -> str | None:
        """Fetch and extract article text from a review URL.

        Uses readability-lxml to extract the main article content,
        stripping navigation, ads, and other page chrome.

        Args:
            url: Review page URL

        Returns:
            Extracted plain text (up to ~2000 chars), or None on failure.
        """
        # Skip AllMusic URLs (TOS prohibits automated access)
        if "allmusic.com" in url:
            logger.info("Skipping AllMusic URL (TOS): %s", url)
            return None

        # SSRF protection: reject private/internal URLs
        if not _is_safe_url(url):
            logger.warning("Rejecting unsafe review URL: %s", url)
            return None

        client = await self._get_client()

        try:
            # Manually follow redirects so we can re-validate each target
            resp = await client.get(url, follow_redirects=False)
            redirects_followed = 0
            while resp.is_redirect and redirects_followed < 5:
                redirect_url = str(resp.next_request.url) if resp.next_request else None
                if not redirect_url or not _is_safe_url(redirect_url):
                    logger.warning("Rejecting unsafe redirect: %s", redirect_url)
                    return None
                resp = await client.get(redirect_url, follow_redirects=False)
                redirects_followed += 1
            resp.raise_for_status()

            doc = ReadableDocument(resp.text)
            # Get readable HTML, then strip tags for plain text
            readable_html = doc.summary()

            # Strip HTML tags to get plain text
            text = re.sub(r"<[^>]+>", " ", readable_html)
            text = re.sub(r"\s+", " ", text).strip()

            if not text:
                return None

            # Truncate to ~2000 chars at a sentence boundary
            if len(text) > 2000:
                cutoff = text.rfind(". ", 1500, 2000)
                if cutoff == -1:
                    cutoff = 2000
                text = text[:cutoff + 1]

            return text
        except Exception as e:
            logger.warning("Review fetch failed for %s: %s", url, e)
            return None

    async def research_album(
        self, artist: str, album: str, full: bool = True, year: int | None = None
    ) -> ResearchData:
        """Run the full research pipeline for an album.

        Args:
            artist: Artist name
            album: Album title
            full: If True, fetch Wikipedia summary too. If False, light research only.
            year: Optional release year to improve search accuracy.

        Returns:
            ResearchData with whatever could be fetched.
        """
        research = ResearchData()

        # Step 1: Search for release group
        rg_mbid = await self.search_album(artist, album, year=year)
        if not rg_mbid:
            return research
        research.musicbrainz_id = rg_mbid

        # Step 2: Look up release group for URLs and earliest release
        rg_data = await self.lookup_release_group(rg_mbid)
        if not rg_data:
            return research

        research.release_date = rg_data.get("release_date")
        research.review_links = rg_data.get("review_urls", [])

        # Step 3: Look up release for track listing, label, credits
        release_mbid = rg_data.get("earliest_release_mbid")
        research.earliest_release_mbid = release_mbid
        if release_mbid:
            release_data = await self.lookup_release(release_mbid)
            if release_data:
                research.track_listing = release_data.get("track_listing", [])
                research.label = release_data.get("label")
                research.credits = release_data.get("credits", {})

        # Step 4: Wikipedia summary (full research only)
        # Try direct Wikipedia URL first, fall back to Wikidata resolution
        wikipedia_url = rg_data.get("wikipedia_url")
        if full and not wikipedia_url and rg_data.get("wikidata_url"):
            wikipedia_url = await self.resolve_wikidata_to_wikipedia(rg_data["wikidata_url"])
        if full and wikipedia_url:
            summary = await self.fetch_wikipedia_summary(wikipedia_url)
            if summary:
                research.wikipedia_summary = summary

        # Step 5: Fetch review texts (full research only)
        review_urls = rg_data.get("review_urls", [])
        if full and review_urls:
            for review_url in review_urls[:2]:
                text = await self.fetch_review_text(review_url)
                if text:
                    research.review_texts.append(text)

        return research

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http and not self._http.is_closed:
            await self._http.aclose()
