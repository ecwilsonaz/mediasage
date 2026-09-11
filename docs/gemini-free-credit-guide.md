# Using Gemini with MediaSage (Free Tier)

Google's Gemini API has a free tier that works well for personal playlist generation. No credit card required.

---

## Setup

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and sign in with any Google account
2. Click **Create API key** and select or create a Google Cloud project
3. Copy the key and add it to your MediaSage config:

```yaml
services:
  mediasage:
    environment:
      - GEMINI_API_KEY=your-api-key-here
```

MediaSage auto-detects Gemini when this key is set.

---

## Free Tier Limits

Free-tier limits are set per model and change over time, so check the current numbers on Google's [rate limits page](https://ai.google.dev/gemini-api/docs/rate-limits). Limits reset at midnight PT.

MediaSage uses two Gemini models by default:

| Role | Default model | Requests per playlist |
|------|---------------|-----------------------|
| Analysis: understand the prompt, write the playlist description | Gemini 3.5 Flash | 2 |
| Generation: pick tracks from the filtered list (the large request) | Gemini 3.5 Flash-Lite | 1 |

So playlists per day are capped by whichever runs out first: half the analysis model's daily limit, or the generation model's daily limit. Album recommendations make a few more requests.

If one model's free quota is much tighter than the other's, point both roles at the same model, or swap them:

```yaml
services:
  mediasage:
    environment:
      - LLM_MODEL_ANALYSIS=gemini-3.5-flash-lite
      - LLM_MODEL_GENERATION=gemini-3.5-flash-lite
```

Current Gemini Flash models have a 1M token context window. Gemini can handle up to ~18,000 tracks per request — far more than other providers.

---

## Cost

On the free tier: **$0.00**.

On the paid tier, a typical playlist costs roughly $0.03–0.25 with the default models; most of that is the Flash-Lite track-selection request.

---

## Things to Know

**Data usage:** On the free tier, Google may use your prompts to improve their products. Enabling billing (even with $0 spent) opts you out.

**Regional restrictions:** The free tier may not be available from the EU, UK, or Switzerland. Setting up billing works around this.

**Rate limits:** 429 errors mean you've hit your limit. Wait a minute or until the daily reset.

---

## Additional Free Options

**$300 Google Cloud credits** — New Google Cloud users can get $300 in free credits (90 days). In [AI Studio](https://aistudio.google.com), go to Settings → Plan information → Set up Billing. Credit card required but not charged until credits run out. At ~$0.03–0.25/playlist, this covers thousands of playlists.

**Google AI Pro credits** — If you pay for Google AI Pro ($19.99/mo), you get $10/month in Cloud credits via the [Google Developer Program](https://developers.google.com/profile).

---

## Quick Reference

| | |
|---|---|
| Get your key | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| Cost | Free (no credit card) |
| Default models | Gemini 3.5 Flash (analysis), Gemini 3.5 Flash-Lite (generation) |
| Playlists/day (free) | Depends on current per-model limits (see above) |
| Context window | 1M tokens |
| Data privacy (free) | Prompts may be used by Google |
| Data privacy (paid) | Prompts not used by Google |
