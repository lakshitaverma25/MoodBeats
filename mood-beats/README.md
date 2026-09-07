# Mood Beats

Show your face, type a sentence, or say it out loud — Mood Beats fuses
whichever of those signals it gets into one mood, then pulls real,
playable song recommendations for it from Spotify. Manual energy /
danceability / valence sliders are still there as a fallback input mode.

## Multi-modal Mood Fusion Engine (new)

Three independent, optional input signals, combined by confidence-weighted
voting in `fusion.py`:

| Signal | How it's captured | How it's scored |
|---|---|---|
| **Face** | Webcam, read live in the browser by [face-api.js](https://github.com/justadudewhohacks/face-api.js) (a small TensorFlow.js CNN loaded from a CDN) | Dominant expression (happy/sad/angry/surprised/fearful/disgusted/neutral) mapped to a mood, weighted by the model's own confidence |
| **Text** | Typed into the text box | `text_sentiment.py` — VADER sentiment (compound score) + a small keyword layer to tell Happy vs Party vs Motivational apart |
| **Voice** | Spoken, transcribed to text in-browser by the Web Speech API | Same scoring as text, applied to the transcript |
| **Manual** (fallback only) | Energy / Danceability / Valence sliders | The original trained RandomForest classifier (`mood_model.pkl`) |

Important privacy/architecture note: the webcam video **never leaves the
browser**. Face detection runs client-side; only the resulting label (e.g.
`"happy", confidence 0.87`) is sent to the server in a hidden form field.
Voice audio is transcribed client-side too — only the transcript text is
sent. This is also why there's no `opencv`/`deepface` server dependency:
the CNN inference happens in the visitor's browser via TensorFlow.js.

Each mood detection — whichever modalities produced it — is logged to a
new `mood_events` table (see `database.py`) and shown on `/history` with a
"Detected via" chip (`face`, `text`, `voice`, `manual`, or a combination
like `face+text`) plus the actual signal (expression / typed text /
transcript / slider values) that led to it.

## What's new in this version

- **Unlimited catalog, not a hardcoded list** — songs now come live from
  Spotify search (`spotify_client.get_mood_songs`), pulled from a pool of
  ~200+ real tracks per mood that refreshes every 12 hours. The old
  24-song hand-typed list only kicks in as an emergency fallback if
  Spotify is unreachable.
- **Real playback + real "listened" tracking** — every song gets an
  embedded Spotify player. `static/js/player.js` uses Spotify's iFrame API
  to detect when a track *actually starts playing* (not just "was shown")
  and logs it to `/log-play`.
- **"Last 3 listened"** on the `/history` page, driven by that real play
  log — separate from the full prediction history below it.
- **Redesigned UI** — slider inputs, a custom music-themed background
  (hand-built SVG: vinyl records + waveform, no stock photo), glass-panel
  cards, hover states.
- **Security/cleanup** — `debug` mode off by default (`FLASK_DEBUG=true`
  to opt in), credentials in `.env`, `pyproject.toml` dropped ~75 unrelated
  notebook/data-science packages.

## Why Spotify enrichment uses *search*, not *recommendations*

Spotify deprecated the `/recommendations` and `/audio-features` endpoints
for any app created after 27 Nov 2024 — new apps get a 403 on those,
permanently. So this app can't ask Spotify "give me tracks with energy 8,
danceability 9". Instead:

1. Your own curated `song_db` in `app.py` still decides *which* songs fit
   each mood (this is the part that actually matters for taste/curation).
2. At request time, `spotify_client.py` uses the **`/search`** endpoint
   (not deprecated) to look up each song by name + artist and pull its
   real cover art and Spotify track ID.
3. Playback uses Spotify's public **embed player** (`open.spotify.com/embed/track/<id>`),
   which is just an iframe — no API call, no dependency on the
   increasingly-unreliable `preview_url` field.
4. Results are cached in SQLite (`spotify_cache` table) so repeat visits
   don't re-hit the Spotify API for the same songs.

## Setup

```bash
pip install -r requirements.txt
# or: uv sync
```

1. Create a Spotify app at https://developer.spotify.com/dashboard
   (any redirect URI works, e.g. `http://127.0.0.1:6767/callback` —
   it's required by the dashboard but unused by this app).
2. Copy `.env.example` to `.env` and fill in `SPOTIFY_CLIENT_ID` /
   `SPOTIFY_CLIENT_SECRET`.
3. (Optional) Put a few audio files in `static/audio/` and cover images
   in `static/image/` — these are only used as a last-resort fallback if
   Spotify search is completely unreachable.
4. Run it:

```bash
python app.py
```

Visit `http://127.0.0.1:6767`.

## Project structure

```
app.py               Flask routes, request handling, fallback song list
fusion.py             Mood Fusion Engine — combines face/text/voice/manual signals
text_sentiment.py     VADER-based text & voice sentiment -> mood
spotify_client.py    Spotify search pool per mood + embed lookups, caching
database.py          SQLite: mood_events (multi-modal), legacy mood_history, play log, Spotify cache
templates/           index.html (face/text/voice/manual predictor), history.html, base.html
static/css/style.css
static/img/music-bg.svg   custom background artwork
static/js/mood_detect.js  webcam facial expression (face-api.js) + Web Speech API voice capture
static/js/player.js       real playback tracking -> /log-play
mood_model.pkl / label_encoder.pkl   trained model used for the manual-slider fallback path
mood_song_dataset.csv                training data (20 rows — see note below)
```

## Tuning which songs show up per mood

Edit `MOOD_QUERIES` at the top of `spotify_client.py` — each mood maps to
a few Spotify search queries (e.g. `"punjabi bhangra party"`). Add,
remove, or make them more specific (a particular artist, decade, or
language) to shift what gets pulled into that mood's pool.

## A note on the model

`mood_song_dataset.csv` currently has 20 rows. It's fine for a demo, but
if you want the mood prediction itself to improve next, that dataset is
the highest-leverage place to invest — more rows, and some borderline
examples between moods (e.g. Motivational vs Party at similar energy),
would help most. Happy to help retrain once you have more data.
