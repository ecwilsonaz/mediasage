# MediaSage for Plex and Jellyfin

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GHCR](https://img.shields.io/badge/ghcr-ryanattanagra%2Fmediasage-blue)](https://github.com/RyanAtTanagra/mediasage/pkgs/container/mediasage)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)

**AI-powered playlists and album recommendations for Plex and Jellyfin—using only music you actually own.**

> Maintained fork of [ecwilsonaz/mediasage](https://github.com/ecwilsonaz/mediasage) with Jellyfin support, current AI models, a model picker in Settings, and fixes from open upstream PRs.

MediaSage is a self-hosted web app that creates playlists and recommends albums by combining LLM intelligence with your Plex or Jellyfin music library. Every suggestion is guaranteed playable because it only considers music you have.

*Sample Generated Playlist:*
![MediaSage Screenshot](docs/images/screenshot-playlist.png)

*Sample Generated Album Recommendation:*
![MediaSage Screenshot](docs/images/screenshot-album.png)

*Home Screen:*
![MediaSage Screenshot](docs/images/screenshot-home.png)

*Playlist Flow:*
![MediaSage Screenshot](docs/images/screenshot-playlist-start.png)

*Album Flow:*
![MediaSage Screenshot](docs/images/screenshot-album-start.png)

---

## Quick Start

```bash
docker run -d \
  --name mediasage \
  -p 5765:5765 \
  -v mediasage-data:/app/data \
  --restart unless-stopped \
  ghcr.io/ryanattanagra/mediasage:latest
```

Open **http://localhost:5765** — a setup wizard walks you through connecting Plex or Jellyfin, choosing an AI provider, and syncing your library.

You can also pass credentials as environment variables to skip the wizard. See [Configuration](#configuration) for details.

**Requirements:** Docker, a Plex server with music and a [Plex token](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/) (or a Jellyfin server with music and a Jellyfin API key), and an API key from Google, Anthropic, or OpenAI (or a local model via Ollama).

---

## Contents

- [Why MediaSage?](#why-mediasage)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Development](#development)
- [API Reference](#api-reference)

---

## Why MediaSage?

**Plex users with personal music libraries have few good options for AI playlists.**

Plexamp's built-in Sonic Sage used ChatGPT to generate playlists, but it was designed around Tidal streaming. The AI recommended tracks from an unlimited catalog, and Tidal made them playable. The "limit to library" setting just hid results you didn't own—so if you asked for 25 tracks and only 4 existed in your library, you got a 4-track playlist.

When [Tidal integration ended in October 2024](https://forums.plex.tv/t/tidal-integration-with-plex-ending-october-28-2024/885728), Sonic Sage lost its foundation. Generic tools like ChatGPT have the same problem: they recommend from an infinite catalog with no awareness of what you actually own.

**MediaSage inverts the approach:**

| Filter-Last (Sonic Sage, ChatGPT) | Filter-First (MediaSage) |
|-----------------------------------|-------------------------|
| AI recommends from infinite catalog | AI only sees your library |
| Hide missing tracks after | No missing tracks possible |
| Near-empty playlists | Full playlists, every time |

The result: every track in every playlist exists in your library and plays immediately.

---

## Features

### Playlist Generation

Create playlists two ways:

**Describe what you want** — Natural language prompts like:
- "Melancholy 90s alternative for a rainy day"
- "Upbeat instrumental jazz for a dinner party"
- "Late night electronic, nothing too aggressive"

**Start from a song** — Pick a track you love, then explore musical dimensions: mood, era, instrumentation, genre, production style. Select which qualities you want more of.

### Album Recommendations

Describe a mood or moment, answer two quick questions about your preferences, and get a single perfect album to listen to—with an editorial pitch explaining why it fits.

- **Library mode** — recommends albums you own, ready for instant playback
- **Discovery mode** — suggests albums you don't own yet, based on your taste profile
- **Familiarity control** — choose between comfort picks, hidden gems, or rediscoveries
- **Show Me Another** — regenerate without starting over
- Primary recommendation with a full write-up, plus two secondary picks

### Smart Filtering

Before the AI sees anything, you control the pool:
- **Genres** — Select from your library's actual genre tags (for Plex, album, artist and track genres combined, so genres you curate on artists count)
- **Decades** — Filter by era
- **Artists** — Only include, or leave out, specific artists. Ask for "a Radiohead playlist" and the artist filter is filled in for you (matched on album artist, so tracks on "Various Artists" compilations aren't included)
- **Minimum rating** — Only tracks rated 3+, 4+, etc. (Plex only; Jellyfin has no star ratings)
- **Exclude live versions** — Skip concert recordings automatically

Real-time track counts show exactly how your filters narrow results.

### Local Library Cache

MediaSage syncs your music library to a local SQLite database. After a one-time sync (about 30 seconds to 2 minutes for 20,000 tracks), all library operations—filtering, counting, sending to AI—happen locally in milliseconds instead of waiting on the media server.

- **Setup wizard** walks you through first-run configuration and sync
- **Footer status** shows track count and last sync time
- **Auto-refresh** re-syncs once the cache is older than the interval in Settings (daily by default; every 6 or 12 hours, weekly, or off)
- **Incremental refresh** (Plex) picks up newly added tracks and tag edits every hour in between, without a full sync. Plays, ratings and deleted tracks are picked up by the next full sync
- **Large libraries** sync in pages (about 10 minutes for ~390,000 tracks); a failed sync never replaces a good cache, and an interrupted one restarts by itself
- **Shrink protection**: an automatic sync that would cut the cache to less than half is treated as a server problem and skipped. If the library really did shrink, refreshing it by hand accepts the smaller library
- **Manual refresh** available anytime, e.g. right after adding music
- **Switching servers** between Plex and Jellyfin, or picking a different music library, re-syncs automatically

### Multi-Provider Support

Bring your own API key—or run locally:

| Provider | Max Tracks | Typical Cost | Best For |
|----------|------------|--------------|----------|
| **Google Gemini** | ~22,000 | $0.03 – $0.25 | Large libraries, lowest cost |
| **Anthropic Claude** | ~4,400 (Haiku) / ~22,000 (Sonnet, Opus) | $0.15 – $1.00 | Nuanced recommendations |
| **OpenAI GPT** | ~23,000 | $0.01 – $0.20 | Solid all-around |
| **Ollama** ⚗️ | Varies | Free | Privacy, local inference |
| **Custom** ⚗️ | Configurable | Free | Self-hosted, OpenAI-compatible APIs |

⚗️ *Local LLM support is experimental. [Report issues](https://github.com/RyanAtTanagra/mediasage/issues).*

> **Free option:** Google Gemini offers a free API tier that's more than enough for personal use — no credit card required. See the [Gemini free credit guide](docs/gemini-free-credit-guide.md) for setup instructions and details.

Estimated cost displays before you generate. MediaSage auto-detects your provider based on which key you configure, and you can pick each provider's models in Settings.

### Play and Save

- **Play Now** — send tracks directly to any Plex device for instant playback (Plex only)
- **Create** a new playlist, **replace** an existing one, or **append** tracks to one
- Device picker shows all active Plex clients with status indicators
- Duplicate detection when appending to existing playlists
- Preview tracks with album art before saving
- Remove tracks you don't want
- Rename the playlist
- See actual token usage and cost

---

## Installation

### Docker Compose (Recommended)

```bash
mkdir mediasage && cd mediasage
curl -O https://raw.githubusercontent.com/RyanAtTanagra/mediasage/main/docker-compose.yml
curl -O https://raw.githubusercontent.com/RyanAtTanagra/mediasage/main/.env.example
mv .env.example .env
```

Start it and open **http://localhost:5765** to run the setup wizard:

```bash
docker compose up -d
```

`.env` is optional. To manage some settings there instead of in the UI, uncomment them in `.env`, for example:

```bash
PLEX_URL=http://your-plex-server:32400
PLEX_TOKEN=your-plex-token
GEMINI_API_KEY=your-gemini-key
# Or for Jellyfin: JELLYFIN_URL, JELLYFIN_TOKEN
```

### NAS Platforms

<details>
<summary><strong>Synology (Container Manager)</strong></summary>

**GUI:**
1. **Container Manager** → **Registry** → Search `ghcr.io/ryanattanagra/mediasage`
2. Download `latest` tag
3. **Container** → **Create**
4. Port: 5765 → 5765
5. Volume: a folder such as `/docker/mediasage/data` → `/app/data`
6. Start it, open port 5765 and complete the setup wizard

**Docker Compose:**
```bash
mkdir -p /volume1/docker/mediasage && cd /volume1/docker/mediasage
curl -O https://raw.githubusercontent.com/RyanAtTanagra/mediasage/main/docker-compose.yml
curl -O https://raw.githubusercontent.com/RyanAtTanagra/mediasage/main/.env.example
mv .env.example .env && nano .env
```
Then in **Container Manager** → **Project** → **Create**, point to `/volume1/docker/mediasage`.

**No Docker?** Some Synology models (especially ARM-based units) don't support Docker/Container Manager. See [Bare Metal](#bare-metal-no-docker) below for running MediaSage directly with Python.

</details>

<details>
<summary><strong>Unraid</strong></summary>

A submission-ready Community Apps template is included at
[`unraid/mediasage.xml`](unraid/mediasage.xml). Until MediaSage is listed in
Community Apps, install it through **Docker → Add Container**:

1. Repository: `ghcr.io/ryanattanagra/mediasage:latest`
2. Port: 5765 → 5765
3. App data: `/mnt/user/appdata/mediasage` → `/app/data`
4. Open the WebUI and complete the setup wizard

The template runs the container as Unraid's standard `nobody:users` account
(`99:100`) so Docker-created appdata is writable.

Maintainers can publish the template by submitting this repository through the
[Unraid Community Apps portal](https://ca.unraid.net/submit). The portal still
requires repository-owner review even when the XML is already present.

</details>

<details>
<summary><strong>TrueNAS SCALE</strong></summary>

1. **Apps** → **Discover Apps** → **Custom App**
2. Image: `ghcr.io/ryanattanagra/mediasage`, Tag: `latest`
3. Port: 5765
4. Storage: a host path or volume mounted at `/app/data` (settings and the library cache live here)
5. Start it, open port 5765 and complete the setup wizard

</details>

<details>
<summary><strong>Portainer</strong></summary>

**Stacks** → **Add Stack**:

```yaml
services:
  mediasage:
    image: ghcr.io/ryanattanagra/mediasage:latest
    ports:
      - "5765:5765"
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

</details>

### Bare Metal (No Docker)

Docker isn't required. MediaSage is Python + FastAPI with no native dependencies, so it runs on any machine with Python 3.11+ — including ARM-based Synology NAS models, Raspberry Pis, or any Linux/macOS/Windows box.

```bash
git clone https://github.com/RyanAtTanagra/mediasage.git
cd mediasage
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Start the server, then complete the setup wizard in your browser:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 5765
```

Access at **http://your-machine-ip:5765**.

<details>
<summary><strong>Running as a background service (systemd)</strong></summary>

To keep MediaSage running after you close your terminal, create a systemd service:

```ini
# /etc/systemd/system/mediasage.service
[Unit]
Description=MediaSage
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/mediasage
EnvironmentFile=/path/to/mediasage/.env
ExecStart=/path/to/mediasage/venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 5765
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable mediasage
sudo systemctl start mediasage
```

</details>

---

## Configuration

### Settings and Setup Wizard

The usual way to configure MediaSage is the setup wizard on first launch, then the **Settings** page. Settings are saved to `config.user.yaml` in the data directory and persist across restarts, so mount `/app/data` as a volume.

### Environment Variables

Environment variables are optional, for scripted or automated installs. A variable with a value overrides what's saved in Settings, and the matching field in Settings is locked with a note naming the variable. Empty variables are ignored, so the compose file's `PLEX_URL=${PLEX_URL:-}`-style lines are harmless when `.env` leaves them unset.

| Variable | Description |
|----------|-------------|
| `MEDIA_SERVER` | `plex` or `jellyfin`. If unset, Jellyfin is used only when `JELLYFIN_URL` is set and `PLEX_URL` isn't |
| `PLEX_URL` | Plex server URL (e.g., `http://192.168.1.100:32400`) |
| `PLEX_PUBLIC_URL` | Plex URL as reached from browsers and players, if different from `PLEX_URL` (e.g. `PLEX_URL=http://plex:32400` inside Docker, `PLEX_PUBLIC_URL=https://plex.example.com`). Used for "Open in Plex" links and for Play Now on devices that can't reach `PLEX_URL` |
| `PLEX_TOKEN` | [Plex authentication token](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/) |
| `PLEX_MUSIC_LIBRARY` | Plex library name if not "Music" |
| `JELLYFIN_URL` | Jellyfin server URL (e.g., `http://192.168.1.100:8096`) |
| `JELLYFIN_TOKEN` | Jellyfin API key |
| `JELLYFIN_MUSIC_LIBRARY` | Jellyfin library name if not "Music" |
| `GEMINI_API_KEY` | Google Gemini API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `LLM_PROVIDER` | Force provider: `gemini`, `anthropic`, `openai`, `ollama`, `custom` |
| `LLM_MODEL_ANALYSIS` / `LLM_MODEL_GENERATION` | Force the analysis or generation model |
| `LLM_TIMEOUT` | Seconds to wait for an AI response (default: 600). Also in Settings as Request Timeout |
| `LIBRARY_SYNC_HOURS` | Re-sync the library cache once it's this many hours old (default: 24; `0` turns automatic re-syncs off). Also in Settings as Auto-Refresh Library |
| `LIBRARY_INCREMENTAL_MINUTES` | Plex: read only added/changed tracks once the cache is this many minutes old (default: 60; `0` turns it off). Plays, ratings and deleted tracks are picked up by the next full sync |
| `LIBRARY_CHECK_SECONDS` | How often the cache is checked for an empty, interrupted, stale or incremental sync (default: 300; `0` turns the periodic check off, leaving one check at startup) |
| `LIBRARY_RETRY_MINUTES` | Wait this long after a failed sync before trying again (default: 30) |
| `PLEX_TIMEOUT` | Seconds to wait for a Plex request (default: 300) |
| `PLEX_SYNC_PAGE_SIZE` | Tracks per Plex request during a sync (default: 5000) |
| `OLLAMA_URL` | Ollama server URL (default: `http://localhost:11434`) |
| `OLLAMA_CONTEXT_WINDOW` | Override detected context window for Ollama (default: 32768) |
| `CUSTOM_LLM_URL` | Custom OpenAI-compatible API base URL |
| `CUSTOM_LLM_API_KEY` | API key for custom provider (if required) |
| `CUSTOM_LLM_MODEL` | Model name for the custom provider, used for analysis and generation |
| `CUSTOM_CONTEXT_WINDOW` | Context window size for custom provider (default: 32768) |

### Jellyfin

Choose **Jellyfin** in the setup wizard or under **Settings → Server Type**, then enter:

- **Server URL**, e.g. `http://192.168.1.100:8096`
- **API key**: create one in Jellyfin under **Dashboard → API Keys**
- **Music library** name (defaults to "Music"; if it isn't found, the first music library is used)

Playlist generation, album recommendations, saving, and replacing or appending to playlists all work as with Plex. A few things differ:

- **No Play Now.** Sending tracks straight to a player is Plex-only, so save a playlist and play it from Jellyfin.
- **No rating filter.** Jellyfin has no star ratings.
- **Genres come from your music's tags**, since Jellyfin's genre index can be empty for music libraries.
- **Playlists** are created as the server's first administrator, since API keys aren't tied to a user. The playlist picker also lists `.m3u` playlists Jellyfin imported from your music folders.

Switching between Plex and Jellyfin re-syncs the library cache automatically.

### Advanced: config.yaml

Mount a config file for additional options:

```yaml
plex:
  music_library: "Music"

llm:
  provider: "gemini"
  model_analysis: "gemini-3.5-flash-lite"
  model_generation: "gemini-3.5-flash-lite"
  smart_generation: false  # true = use smarter model for both (higher quality, ~3-5x cost)

defaults:
  track_count: 25
```

### Model Selection

MediaSage uses a two-model strategy by default:

| Role | Purpose | Models Used |
|------|---------|-------------|
| **Analysis** | Interpret prompts, suggest filters, analyze seed tracks | claude-sonnet-5-5 / gpt-6.1-sol / gemini-3.5-flash-lite |
| **Generation** | Select tracks from filtered list | claude-haiku-4-5 / gpt-6-luna / gemini-3.5-flash-lite |

This balances quality with cost. Choose other models in Settings, or with `LLM_MODEL_ANALYSIS` / `LLM_MODEL_GENERATION` (which override Settings). For Claude, a 1M-context generation model such as `claude-sonnet-5-5` fits ~5x more tracks than Haiku at about twice the per-token price. Enable `smart_generation: true` to use the analysis model for everything.

### Local LLM Setup (Experimental)

Run MediaSage with local models for privacy and zero API costs.

<details>
<summary><strong>Ollama</strong></summary>

1. Install [Ollama](https://ollama.ai) and pull a model:
   ```bash
   ollama pull llama3:8b
   ```

2. Configure MediaSage in Settings, or with environment variables:
   ```bash
   LLM_PROVIDER=ollama
   OLLAMA_URL=http://localhost:11434
   ```

3. Select your model in Settings → the context window is auto-detected.

**Recommended models:** `llama3:8b`, `qwen3:8b`, `mistral` — models with 8K+ context work best.

</details>

<details>
<summary><strong>Custom OpenAI-Compatible API</strong></summary>

For LM Studio, text-generation-webui, vLLM, or any OpenAI-compatible service (OpenRouter, Mammouth, etc.):

1. Start your server, or get the service's base URL and API key

2. Configure in the setup wizard or Settings (or with `CUSTOM_LLM_URL`, `CUSTOM_LLM_API_KEY` and `CUSTOM_LLM_MODEL`):
   - **API Base URL:** `http://localhost:5000/v1`
   - **API Key:** If required by your server
   - **Model Name:** The model identifier (required; the service's `/v1/models` lists them)
   - **Context Window:** Your model's context size (Settings only)

</details>

**Note:** Local models are slower and may produce less accurate results than cloud providers. MediaSage waits 10 minutes for a response by default; on slow hardware, raise **Request Timeout** in Settings or send fewer tracks to the AI. Models with larger context windows will support more tracks.

---

## How It Works

MediaSage uses a **filter-first architecture** designed for large libraries (50,000+ tracks):

```
┌─────────────────────────────────────────────────────────────────┐
│  1. ANALYZE                                                      │
│     LLM interprets your prompt → suggests genre/decade filters   │
├─────────────────────────────────────────────────────────────────┤
│  2. FILTER                                                       │
│     Library narrowed to matching tracks                          │
│     "90s Alternative" → 2,000 tracks                             │
├─────────────────────────────────────────────────────────────────┤
│  3. SAMPLE                                                       │
│     If too large for context, randomly sample                    │
│     Fits within model's token limits                             │
├─────────────────────────────────────────────────────────────────┤
│  4. GENERATE                                                     │
│     Filtered track list + prompt sent to LLM                     │
│     LLM selects best matches from available tracks               │
├─────────────────────────────────────────────────────────────────┤
│  5. MATCH                                                        │
│     Fuzzy matching links LLM selections to library               │
│     Handles minor spelling/formatting differences                │
├─────────────────────────────────────────────────────────────────┤
│  6. SAVE                                                         │
│     Playlist created in Plex or Jellyfin                         │
│     Ready in Plexamp, Jellyfin, or any client                    │
└─────────────────────────────────────────────────────────────────┘
```

This ensures every track exists in your library while keeping API costs manageable.

---

## Development

### Local Setup

```bash
git clone https://github.com/RyanAtTanagra/mediasage.git
cd mediasage
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export PLEX_URL=http://your-plex-server:32400
export PLEX_TOKEN=your-plex-token
export GEMINI_API_KEY=your-key

uvicorn backend.main:app --reload --port 5765
```

### Testing

```bash
pytest tests/ -v
```

### Tech Stack

- **Backend:** Python 3.11+, FastAPI, python-plexapi, httpx (Jellyfin REST API), rapidfuzz
- **Frontend:** Vanilla HTML/CSS/JS (no build step)
- **LLM SDKs:** anthropic, openai, google-genai (+ Ollama via REST API)
- **Deployment:** Docker

---

## API Reference

Interactive documentation available at `/docs` when running.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/config` | GET/POST | Get or update configuration |
| `/api/setup/status` | GET | Onboarding checklist state |
| `/api/setup/validate-plex` | POST | Validate Plex credentials |
| `/api/setup/validate-jellyfin` | POST | Validate Jellyfin credentials |
| `/api/setup/validate-ai` | POST | Validate AI provider credentials |
| `/api/setup/complete` | POST | Mark setup wizard as complete |
| `/api/library/stats` | GET | Library statistics |
| `/api/library/status` | GET | Cache state, track count, sync progress, last attempt/error, incremental watermark, interrupted flag |
| `/api/library/sync` | POST | Trigger background library sync (`?incremental=true`: only added/changed tracks, Plex; `?force=true`: accept a library less than half the size of the cache) |
| `/api/library/search` | GET | Search library tracks |
| `/api/library/artists` | GET | Search library artists (for the artist filter) |
| `/api/analyze/prompt` | POST | Analyze natural language prompt |
| `/api/analyze/track` | POST | Analyze a seed track |
| `/api/filter/preview` | POST | Preview filtered track list |
| `/api/generate/stream` | POST | Stream playlist generation (SSE) |
| `/api/playlist` | POST | Save playlist to Plex or Jellyfin |
| `/api/playlist/update` | POST | Replace or append to a playlist |
| `/api/recommend/albums/preview` | GET | Preview album candidates for filters |
| `/api/recommend/analyze-prompt` | POST | Analyze prompt for genre/decade filters |
| `/api/recommend/questions` | POST | Generate clarifying questions |
| `/api/recommend/generate` | POST | Generate album recommendations |
| `/api/recommend/switch-mode` | POST | Switch library/discovery mode |
| `/api/results` | GET | List saved result history |
| `/api/results/{id}` | GET/DELETE | Get or delete a saved result |
| `/api/plex/clients` | GET | List active Plex clients |
| `/api/plex/playlists` | GET | List existing Plex playlists |
| `/api/jellyfin/playlists` | GET | List existing Jellyfin playlists |
| `/api/play-queue` | POST | Send tracks to a Plex client |
| `/api/art/{rating_key}` | GET | Proxy album art from Plex or Jellyfin |
| `/api/models` | GET | Cloud AI models offered in Settings, with prices |
| `/api/ollama/status` | GET | Ollama connection status |
| `/api/ollama/models` | GET | List available Ollama models |
| `/api/ollama/model-info` | GET | Get model details (context window) |

---

## License

MIT
