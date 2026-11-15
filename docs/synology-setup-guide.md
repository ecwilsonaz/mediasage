# MediaSage on Synology DiskStation — Setup Guide

This guide shows you how to install MediaSage on a Synology NAS. No command-line experience required.

---

## What You Need

Before you start, gather these three things:

1. **A Synology NAS with Container Manager installed.** Container Manager is a free app in Synology's Package Center (it used to be called "Docker"). If it's missing, open Package Center, search "Container Manager," and install it.

   > **Don't have Container Manager?** Some Synology models (especially ARM-based units like the DS220j or DS223j) don't support it. You can still run MediaSage directly with Python — see the [Bare Metal install instructions](../README.md#bare-metal-no-docker) in the main README.

2. **Your Plex token.** This lets MediaSage connect to your Plex server. See [Finding Your Plex Token](#finding-your-plex-token) below.

3. **A Gemini API key (free).** MediaSage uses an AI service to build playlists. Google Gemini is free and requires no credit card. See [Getting a Gemini API Key](#getting-a-gemini-api-key-free) below.

---

## Finding Your Plex Token

Your Plex token is a long string that lets other apps talk to your Plex server. Here's how to find it:

1. Open Plex in a web browser and sign in
2. Browse to any media item (a song, movie, or show)
3. Click the **three dots** (⋯) menu and choose **Get Info**
4. Click **View XML** in the bottom-left corner
5. A new tab opens with XML text. Look at the URL in your browser's address bar — at the end you'll see `X-Plex-Token=xxxxxxxxxxxxxxxxxxxx`
6. Copy everything after `X-Plex-Token=` — that's your token

Plex maintains an [official guide for finding your token](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/).

> **Keep your token private.** Anyone with your token can access your Plex server.

---

## Getting a Gemini API Key (Free)

Google Gemini is the recommended AI provider for MediaSage. It's free for personal use, handles the largest music libraries, and requires no credit card.

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with any Google account
3. Click **Create API key**
4. If prompted, select or create a Google Cloud project (any name works)
5. Copy the key that appears

The free tier gives you roughly 100-500 playlists per day — more than enough.

> **Note:** On the free tier, Google may use your prompts to improve their products. Your music library data (artist names, track titles) is included in prompts. If this concerns you, enable billing (even with $0 spent) to opt out. See the [Gemini free credit guide](gemini-free-credit-guide.md) for details.

---

## Setting Up MediaSage

Two methods exist. **Method A (Docker Compose)** is recommended because updates are easier. Method B uses the GUI only.

Both methods require your Synology user ID so the container can write to its data folder securely. Find it first.

### Finding Your Synology User ID

MediaSage stores its library cache and settings (which may include API keys) in a data folder. To keep this folder secure, the container runs as *your* Synology user instead of granting broad access. You need two numbers: your **UID** and **GID**.

1. Open **Control Panel** → **Task Scheduler**
2. Click **Create** → **Scheduled Task** → **User-defined script**
3. **General** tab: Name it `Find my UID` and set **User** to your admin account
4. **Schedule** tab: Set **Run on the following date**, pick today, and set it to not repeat
5. **Task Settings** tab: In the **User-defined script** box, paste:
   ```
   id > /volume1/docker/mediasage/myid.txt
   ```
6. Click **OK** (dismiss any warnings)
7. Select the task and click **Run** → **Yes**
8. Open **File Station** → `docker/mediasage` → double-click `myid.txt`

You'll see something like:

```
uid=1026(admin) gid=100(users) groups=100(users),101(administrators)
```

The two numbers you need are `1026` (your UID) and `100` (your GID). Write them down.

9. Back in Task Scheduler, select the task and **Delete** it (it was only needed once)

> **Note:** This step requires the `docker/mediasage` folder to exist first. If you haven't created it yet, see Step 1 below, then come back here.

---

### Method A: Docker Compose Project (Recommended)

This method creates a "Project" in Container Manager using a compose file. It handles the data folder and settings in one place.

#### Step 1: Create the project folder

1. Open **File Station** on your Synology
2. Navigate to the `docker` shared folder
   - If it's missing, create a shared folder called `docker` (Control Panel → Shared Folder → Create)
3. Inside `docker`, create a folder called `mediasage`
4. Inside `mediasage`, create a folder called `data`

Your folder structure should look like:
```
docker/
└── mediasage/
    └── data/
```

#### Step 2: Create the compose file

1. On your computer, open a text editor (Notepad on Windows, TextEdit on Mac set to plain text)
2. Paste the following — you'll replace the values in angle brackets `< >` with your own:

> **The `environment` lines are optional.** You can delete them (and the `environment:` line) and enter your Plex and AI details in the setup wizard the first time you open MediaSage instead. Settings saved there are kept in the `data` folder. If you do set them here, they take priority over Settings, and the matching fields in Settings are locked.

```yaml
services:
  mediasage:
    image: ghcr.io/ecwilsonaz/mediasage:latest
    container_name: mediasage
    user: "<UID>:<GID>"                            # Your Synology user and group ID
    ports:
      - "5765:5765"
    environment:
      - PLEX_URL=http://<PLEX_IP>:32400            # Your Plex server's IP address
      - PLEX_TOKEN=<PLEX_TOKEN>                    # Your Plex token (long string)
      - GEMINI_API_KEY=<GEMINI_KEY>                # Your Gemini API key
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

**Replace each `<placeholder>` with your actual value:**

| Placeholder | Replace with | Example |
|---|---|---|
| `<UID>` | Your Synology user ID number | `1026` |
| `<GID>` | Your Synology group ID number | `100` |
| `<PLEX_IP>` | Your Plex server's local IP address | `192.168.1.50` |
| `<PLEX_TOKEN>` | Your Plex authentication token | `A1b2C3d4E5f6G7h8I9j0` |
| `<GEMINI_KEY>` | Your Gemini API key from Google AI Studio | `AIzaSyB-1234abcd5678efgh` |

See [Finding Your Synology User ID](#finding-your-synology-user-id) for UID/GID and [Finding Your Plex Server's IP](#finding-your-plex-servers-ip) for the Plex IP.

> **Important:** Replace the placeholder *including* the angle brackets `< >`. For example, `<UID>` becomes `1026`, not `<1026>`. Everything to the left of the `=` sign (like `PLEX_TOKEN=`) must stay exactly as shown — only change what's to the right.

**Example with real values** (yours will be different):

```yaml
services:
  mediasage:
    image: ghcr.io/ecwilsonaz/mediasage:latest
    container_name: mediasage
    user: "1026:100"
    ports:
      - "5765:5765"
    environment:
      - PLEX_URL=http://192.168.1.50:32400
      - PLEX_TOKEN=A1b2C3d4E5f6G7h8I9j0
      - GEMINI_API_KEY=AIzaSyB-1234abcd5678efgh
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

3. **Verify your file before saving:**
   - The `user:` line has two numbers separated by a colon (e.g., `"1026:100"`) — no letters
   - Each environment line has a variable name, then `=`, then your value (e.g., `PLEX_TOKEN=A1b2...`)
   - No angle brackets `< >` remain anywhere in the file

4. Save the file as `docker-compose.yml`
5. In **File Station**, upload this file into the `docker/mediasage` folder you created

Your folder should now look like:
```
docker/
└── mediasage/
    ├── docker-compose.yml
    └── data/
```

#### Step 3: Create the project in Container Manager

1. Open **Container Manager** on your Synology
2. Go to **Project** in the left sidebar
3. Click **Create**
4. Set the **Project Name** to `mediasage`
5. Set the **Path** to `/volume1/docker/mediasage` (or wherever your folder is)
6. Container Manager should detect the `docker-compose.yml` automatically
7. Click **Next**, then **Next** again (skip the web portal setup)
8. Click **Done**

Container Manager will download the MediaSage image and start the container. The first download takes a minute or two.

#### Step 4: Verify it's running

1. In Container Manager → **Project**, you should see `mediasage` with a green "Running" status
2. Open your browser and go to `http://<your-synology-ip>:5765`
3. Click **Settings** in the top navigation and confirm:
   - **Plex Connection** shows a green status with your library name and track count
   - **LLM Provider** shows a green status with "Gemini" configured
4. If the container isn't running, see [Troubleshooting](#troubleshooting). If the container is green but Plex or the LLM shows disconnected in Settings, see [Container runs but Plex or LLM shows disconnected](#container-runs-but-plex-or-llm-shows-disconnected)

---

### Method B: Container Manager GUI Only

If you prefer the GUI to editing a compose file, use this method.

#### Step 1: Create the data folder

1. Open **File Station**
2. Navigate to the `docker` shared folder (create one if needed)
3. Create a folder called `mediasage`
4. Inside `mediasage`, create a folder called `data`

#### Step 2: Download the image

1. Open **Container Manager**
2. Go to **Registry** in the left sidebar
3. The image is on GitHub's registry, not Docker Hub, so add it once: **Settings** → **Add**, name `GHCR`, URL `https://ghcr.io`, then select it and click **Use**
4. In the search bar, type `ecwilsonaz/mediasage` and click **Download**
5. Choose the `latest` tag and click **Apply**
6. Wait for the download to complete (check the **Image** section)

#### Step 3: Create the container

1. Go to **Image** in the left sidebar
2. Select the `mediasage` image and click **Run**
3. Set **Container Name** to `mediasage`
4. Check **Enable auto-restart**
5. Under **Execution Settings** (if available), check **Execute container using high privilege** is OFF, and set **User** to your UID (e.g., `1026`) and **Group** to your GID (e.g., `100`) from [Finding Your Synology User ID](#finding-your-synology-user-id)
6. Click **Next**

> **If you don't see Execution Settings:** Your DSM version may not support it in the GUI. After setup, if you get a "permission denied" error, see [Troubleshooting](#troubleshooting) for an alternative fix.

#### Step 4: Configure port

1. Under **Port Settings**, set:
   - Local Port: `5765`
   - Container Port: `5765`
   - Protocol: TCP
2. If `5765` is already in use, pick another number for Local Port (e.g., `5766`) but keep Container Port as `5765`

#### Step 5: Configure the data folder

1. Under **Volume Settings**, click **Add Folder**
2. Browse to the `docker/mediasage/data` folder you created
3. Set the **Mount Path** to `/app/data`

#### Step 6: Add environment variables (optional)

You can skip this step and enter these details in the setup wizard when you first open MediaSage. If you add them here, they take priority over Settings. Under **Environment**, add these three variables:

| Variable Name | Value |
|---|---|
| `PLEX_URL` | `http://<PLEX_IP>:32400` (e.g., `http://192.168.1.50:32400`) |
| `PLEX_TOKEN` | Your Plex token (long string of letters and numbers) |
| `GEMINI_API_KEY` | Your Gemini API key from Google AI Studio |

Replace `<PLEX_IP>` with your Plex server's local IP address.

#### Step 7: Start the container

1. Click **Next** to review your settings
2. Click **Done**
3. The container starts automatically

---

## Accessing MediaSage

Once the container runs:

1. Open a web browser on any device connected to your home network
2. Go to: `http://<SYNOLOGY_IP>:5765`

Replace `<SYNOLOGY_IP>` with your Synology's local IP address (e.g., `http://192.168.1.100:5765`).

> **Important:** Use your Synology's IP address, not `localhost` or `127.0.0.1`. Those work only when the app runs on the same machine as your browser. MediaSage runs on your Synology, so you must use the Synology's IP address.

### Finding Your Synology's IP Address

You can find your Synology's IP address in several ways:

- **On your Synology:** Control Panel → Network → Network Interface — look for the IP next to your active connection (usually `LAN` or `Bond`)
- **On your router:** Check connected devices for your DiskStation
- **In DSM:** The IP appears in the browser address bar when you're logged into your Synology's web interface

---

## First-Time Setup

When you first open MediaSage, a setup wizard connects to Plex (or Jellyfin) and your AI provider; steps you already configured with environment variables are filled in. It then syncs your music library, which builds a local index so the app can work with your tracks quickly.

1. **Library sync starts automatically.** A progress bar shows while it scans your library. This takes about 1-2 minutes for a typical library (subsequent visits are faster since the cache is stored in the `data` folder).

2. **Check the Settings page.** Click **Settings** in the top navigation. You should see:
   - **Plex Connection:** Green status showing your library name and track count
   - **LLM Provider:** Green status showing Gemini is configured

If either shows red or unconfigured, check the details in Settings, or your environment variables if you set them (see [Troubleshooting](#troubleshooting)).

---

## Creating Your First Playlist

1. On the main screen, type a prompt describing the music you want. For example:
   - "Chill 90s alternative for a rainy afternoon"
   - "Upbeat classic rock road trip songs"
   - "Jazz instrumentals for cooking dinner"

2. Click **Analyze**. MediaSage suggests genre and decade filters based on your prompt.

3. Adjust the filters if you want. The track count updates in real time as you narrow or expand the pool.

4. Click **Generate Playlist**. The AI picks tracks from your filtered library.

5. Review the results. You can remove tracks you don't want, rename the playlist, and see the estimated cost (usually $0.00 on Gemini's free tier).

6. Click **Save to Plex** to create the playlist. It appears immediately in Plexamp and any other Plex client.

---

## Updating MediaSage

When a new version releases:

**If you used Method A (Docker Compose):**
1. Open **Container Manager** → **Project**
2. Select `mediasage`
3. Click **Action** → **Build** (this pulls the latest image and recreates the container)

**If you used Method B (GUI):**
1. Stop the container: **Container** → select `mediasage` → **Action** → **Stop**
2. Delete the container (this preserves your data): **Action** → **Delete**
3. Delete the old image: **Image** → select the mediasage image → **Delete**
4. Re-download the image from **Registry** (same steps as before)
5. Re-create the container with the same settings

Your library cache and settings are stored in the `data` folder and survive updates.

---

## Troubleshooting

### "This site can't be reached" / page won't load

- **Are you using the right IP?** You need your **Synology's** IP address, not `localhost`. See [Finding Your Synology's IP Address](#finding-your-synologys-ip-address).
- **Is the container running?** Open Container Manager → Container (or Project) and verify mediasage shows a green "Running" status.
- **Is the port correct?** Use port `5765` in the URL (e.g., `http://192.168.1.100:5765`). If you changed the local port during setup, use that number instead.
- **Firewall?** Synology's built-in firewall might block the port. Check Control Panel → Security → Firewall.

### "Permission denied writing to config.user.yaml"

The container cannot write to its data directory. This happens when the container's user doesn't match the folder's owner.

**If you used Method A (Docker Compose):** Verify the `user:` line in your `docker-compose.yml` matches your Synology UID and GID. See [Finding Your Synology User ID](#finding-your-synology-user-id).

**If you used Method B (GUI) and couldn't set the user:** Grant your admin account write access to the data folder:

1. Open **File Station**
2. Navigate to `docker/mediasage/data`
3. Right-click the `data` folder → **Properties**
4. Go to the **Permission** tab
5. Click **Create**, select your admin user, and grant **Read & Write** access
6. Click **Done**, then **Save**
7. Restart the container in Container Manager

If that still fails, the container may be running as a different user than your admin account. The most reliable fix is to switch to Method A (Docker Compose), which lets you set the container's user explicitly.

### "Plex connection failed" / Plex shows as disconnected

- **Check the IP address.** `PLEX_URL` must be an IP address your Synology can reach. If Plex runs on a different machine, use that machine's local IP (e.g., `http://192.168.1.50:32400`), not `localhost`.
- **Is Plex on the same network?** If your Synology and Plex server are on the same local network, use the local IP. If Plex is remote, you may need the external URL.
- **VPN complications.** If either your Synology or Plex server uses a VPN, connections between them may be blocked. Verify each can reach the other — try accessing your Plex server's web UI from your Synology's browser (if available).
- **Check the port.** The default Plex port is `32400`. If you changed it, update `PLEX_URL` accordingly.
- **Verify the token.** An invalid token fails silently. Re-check the token using the [official guide](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/).

### "LLM not configured" / LLM shows as disconnected

- **Check your API key.** Open Container Manager, find the mediasage container, and verify the `GEMINI_API_KEY` environment variable is set and contains no extra spaces.
- **Regenerate the key.** Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and create a new key. Replace the old one in your environment variables, then restart the container.

### Container runs but Plex or LLM shows disconnected

If the container starts successfully (green status in Container Manager) but the Settings page shows Plex or the LLM as disconnected, your `docker-compose.yml` likely has a formatting mistake. Open the file and check for these common issues:

**Variable name was deleted.** Each environment line needs the variable name, an `=` sign, then your value:
```
Wrong:  - A1b2C3d4E5f6G7h8I9j0
Right:  - PLEX_TOKEN=A1b2C3d4E5f6G7h8I9j0
```

**Angle brackets were left in.** Replace the entire placeholder including the `< >`:
```
Wrong:  - PLEX_TOKEN=<A1b2C3d4E5f6G7h8I9j0>
Right:  - PLEX_TOKEN=A1b2C3d4E5f6G7h8I9j0
```

**Extra spaces around the `=` sign.** YAML environment variables cannot have spaces around `=`:
```
Wrong:  - PLEX_TOKEN = A1b2C3d4E5f6G7h8I9j0
Right:  - PLEX_TOKEN=A1b2C3d4E5f6G7h8I9j0
```

**UID/GID still contains letters.** The `user:` line should have only numbers:
```
Wrong:  user: "1026:YOUR-100"
Right:  user: "1026:100"
```

After fixing, re-upload `docker-compose.yml` and rebuild the project: Container Manager → **Project** → select `mediasage` → **Action** → **Build**.

### Library sync is slow or stuck

- First sync takes 1-2 minutes for most libraries. Libraries with 50,000+ tracks may take longer.
- If it seems stuck, check the container logs: Container Manager → Container → select `mediasage` → **Log**. Look for error messages.
- Verify your Plex server is online and reachable.

### Container won't start / shows error on launch

- Check the logs: Container Manager → Container → select `mediasage` → **Log**
- If you see "port already in use," another app is using port 5765. Change the local port in your container settings to something else (e.g., `5766`).
- If you see memory errors, your NAS may lack sufficient RAM. MediaSage is lightweight but needs at least 256 MB free.

---

## Finding Your Plex Server's IP

MediaSage must reach your Plex server over the network. What you enter for `PLEX_URL` depends on where Plex runs:

| Plex runs on... | PLEX_URL value |
|---|---|
| The same Synology NAS | `http://<SYNOLOGY_IP>:32400` (not `localhost` — this fails from inside a container) |
| Another computer on your LAN | `http://<THAT_COMPUTERS_IP>:32400` |
| A remote server | Your Plex server's external URL or IP |

To find a computer's local IP:
- **Windows:** Open Command Prompt, type `ipconfig`, look for "IPv4 Address"
- **Mac:** System Settings → Network → click your connection → IP address
- **Linux:** Open a terminal, type `hostname -I`

> **Use the actual IP address for PLEX_URL, not `localhost` or `127.0.0.1`.** Inside a Docker container, `localhost` refers to the container itself, not your NAS or your computer.

---

## Using a Different AI Provider

MediaSage works with several AI providers. Gemini is the default because it's free and handles the largest libraries, but you can use others:

| Provider | Environment Variable | Cost | Notes |
|---|---|---|---|
| **Google Gemini** | `GEMINI_API_KEY` | Free tier available | Recommended. Handles ~22,000 tracks. |
| **OpenAI** | `OPENAI_API_KEY` | ~$0.01-0.20/playlist | Handles ~23,000 tracks. |
| **Anthropic Claude** | `ANTHROPIC_API_KEY` | ~$0.15-1.00/playlist | Handles ~4,400 tracks with Haiku, ~22,000 with Sonnet. Nuanced taste. |

The simplest way to switch is **Settings → LLM Provider**: pick the provider, paste its key and save. Each provider keeps its own key, so you can switch back and forth.

Or add the appropriate environment variable to your container. MediaSage auto-detects which provider to use based on which key is set. If you set multiple keys, it defaults to Gemini; set `LLM_PROVIDER` explicitly to choose (e.g., `LLM_PROVIDER=openai`).

---

## Optional: Accessing MediaSage Outside Your Home

By default, MediaSage is accessible only on your local network. To access it remotely:

- **Synology QuickConnect** does not work with custom Docker containers.
- **Reverse proxy (advanced):** Set up a reverse proxy in Synology's Control Panel → Login Portal → Advanced → Reverse Proxy. Point a subdomain to `localhost:5765`. This requires a domain name and HTTPS certificate.
- **Tailscale / VPN:** The simplest option. Install Tailscale on your Synology (available in Package Center) and your devices. Access MediaSage via your Synology's Tailscale IP. No port forwarding or domain required.
