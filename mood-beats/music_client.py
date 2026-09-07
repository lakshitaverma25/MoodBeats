"""
music_client.py
----------------
Free, key-less replacement for spotify_client.py.

Spotify tightened Developer Mode in Feb/Mar 2026: the app owner now needs
an active Spotify Premium subscription just to use it, and new apps are
capped at 5 users without a lengthy extended-access review. That's a
non-starter for a student/demo project, so this module gets real track
metadata + real playable 30-second previews from two APIs that need
NO signup, NO API key, and NO subscription:

  1. Deezer Search API   (primary)  - https://api.deezer.com/search
  2. iTunes Search API   (fallback) - https://itunes.apple.com/search
     used only for a track Deezer's search didn't find.

Both return a direct, playable preview URL (mp3 for Deezer, m4a for
iTunes) — unlike Spotify's own API, which stopped returning preview_url
for most tracks years ago. That direct URL means we no longer need any
iframe/embed player at all: every song can just use a plain HTML5
<audio> tag with that URL as its src.

If both APIs fail (e.g. no internet), functions here return an empty
pool / None, and app.py's existing local FALLBACK_SONG_DB path kicks in
exactly as it did before.
"""

import logging
import requests

import database

logger = logging.getLogger(__name__)

DEEZER_SEARCH_URL = "https://api.deezer.com/search"
ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
REQUEST_TIMEOUT = 6  # seconds — fail fast rather than hang a page load

# Search queries used to build each mood's track pool. Several queries per
# mood so the pool has real variety instead of being a fixed hand-picked
# list.
MOOD_QUERIES = {
    "Happy": [
        "bollywood happy songs",
        "punjabi happy upbeat",
        "hindi feel good song",
    ],
    "Sad": [
        "bollywood sad songs",
        "punjabi sad songs",
        "hindi heartbreak song",
    ],
    "Motivational": [
        "hindi motivational rap",
        "punjabi motivational song",
        "bollywood inspirational anthem",
    ],
    "Party": [
        "bollywood party songs",
        "punjabi bhangra party",
        "hindi dance club song",
    ],
}

POOL_SIZE_PER_QUERY = 25


def is_configured():
    """No credentials needed for either API — always usable."""
    return True


def _deezer_track_to_dict(track):
    album = track.get("album") or {}
    artist = track.get("artist") or {}
    return {
        "name": track.get("title"),
        "artist": artist.get("name"),
        "track_id": f"deezer-{track.get('id')}",
        "image_url": album.get("cover_big") or album.get("cover_medium"),
        "spotify_url": track.get("link"),  # kept as "spotify_url" for
                                            # compatibility with existing
                                            # templates/db columns — it's
                                            # really just "source page url"
        "audio_url": track.get("preview"),
        "embed_url": None,
    }


def _itunes_track_to_dict(track):
    artwork = track.get("artworkUrl100") or ""
    # iTunes only gives a 100x100 thumbnail by default; asking for a
    # bigger size is just a URL string swap.
    artwork_big = artwork.replace("100x100bb", "512x512bb") if artwork else None
    return {
        "name": track.get("trackName"),
        "artist": track.get("artistName"),
        "track_id": f"itunes-{track.get('trackId')}",
        "image_url": artwork_big,
        "spotify_url": track.get("trackViewUrl"),
        "audio_url": track.get("previewUrl"),
        "embed_url": None,
    }


def _search_deezer(query, limit=25):
    try:
        resp = requests.get(
            DEEZER_SEARCH_URL,
            params={"q": query, "limit": limit},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        tracks = []
        for item in data.get("data", []):
            if item.get("preview"):  # skip tracks with no playable preview
                tracks.append(_deezer_track_to_dict(item))
        return tracks
    except requests.RequestException as exc:
        logger.error(f"Deezer search failed for '{query}': {exc}")
        return []


def _search_itunes(query, limit=25, country="IN"):
    try:
        resp = requests.get(
            ITUNES_SEARCH_URL,
            params={
                "term": query,
                "entity": "song",
                "limit": limit,
                "country": country,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        tracks = []
        for item in data.get("results", []):
            if item.get("previewUrl"):
                tracks.append(_itunes_track_to_dict(item))
        return tracks
    except requests.RequestException as exc:
        logger.error(f"iTunes search failed for '{query}': {exc}")
        return []


def enrich_song(name, artist):
    """
    Look up a single song's real metadata + preview, using the local
    cache first. Returns None if not found on either provider — callers
    should fall back to a local static asset in that case.
    """
    cached = database.get_cached_track(name, artist)
    if cached is not None:
        if not cached["found"]:
            return None
        return {
            "track_id": cached["track_id"],
            "image_url": cached["image_url"],
            "spotify_url": cached["spotify_url"],
            "audio_url": cached["preview_url"],
            "embed_url": None,
        }

    query = f'track:"{name}" artist:"{artist}"'
    results = _search_deezer(query, limit=1) or _search_itunes(f"{name} {artist}", limit=1)

    if not results:
        database.set_cached_track(name, artist, None, None, None, None, found=False)
        return None

    track = results[0]
    database.set_cached_track(
        name, artist, track["track_id"], track["image_url"],
        track["spotify_url"], track["audio_url"], found=True,
    )
    return track


def _fetch_mood_pool(mood):
    """Pull a fresh pool of real tracks (with playable previews) for a mood."""
    seen_keys = set()
    pool = []
    for query in MOOD_QUERIES.get(mood, []):
        tracks = _search_deezer(query, limit=POOL_SIZE_PER_QUERY)
        if not tracks:
            # Deezer unreachable/empty for this query — try iTunes instead.
            tracks = _search_itunes(query, limit=POOL_SIZE_PER_QUERY)
        for track in tracks:
            key = (track["name"], track["artist"])
            if not track["name"] or key in seen_keys:
                continue
            seen_keys.add(key)
            pool.append(track)

    return pool


def get_mood_songs(mood, count=6):
    """
    Return `count` real songs for a mood, drawn from a cached search pool
    (refreshed periodically). Returns None if nothing could be fetched at
    all — the caller should fall back to the small built-in list only then.
    """
    import random

    pool = database.get_mood_pool(mood)
    if pool is None:
        pool = _fetch_mood_pool(mood)
        if pool:
            database.set_mood_pool(mood, pool)

    if not pool:
        return None

    return random.sample(pool, min(count, len(pool)))
