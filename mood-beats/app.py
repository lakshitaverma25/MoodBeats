import os
import random
import logging
import warnings

from flask import (
    Flask,
    render_template,
    request,
    url_for,
    send_from_directory,
    send_file,
    jsonify,
)
import joblib
from dotenv import load_dotenv

import database
import music_client
import fusion

load_dotenv()  # reads .env if present; real env vars always win

# The model was trained on a DataFrame with named columns (Energy,
# Danceability, Valence). We predict from plain lists for simplicity, which
# is functionally identical but makes sklearn warn about missing feature
# names on every request — silence just that specific, harmless warning.
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning
)

app = Flask(__name__, static_folder="static", static_url_path="/static")

DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model and label encoder
model = joblib.load("mood_model.pkl")
le = joblib.load("label_encoder.pkl")

database.init_db()

# Emergency fallback ONLY — used if a live Deezer/iTunes search totally
# fails (e.g. no internet). The normal path pulls songs live from Deezer
# search (music_client.get_mood_songs), so the catalog isn't capped at
# this short hand-picked list.
FALLBACK_SONG_DB = {
    "Happy": [
        {"name": "Yaarian (ABCD)", "artist": "Yo Yo Honey Singh", "img": "happy1.jpg", "file": "happy1.mp3"},
        {"name": "Tere Te", "artist": "Guru Randhawa", "img": "happy2.jpg", "file": "happy2.mp3"},
        {"name": "Jatt Diyan Tauran", "artist": "Gippy Grewal, Shipra Goyal", "img": "happy3.jpg", "file": "happy3.mp3"},
        {"name": "Jeene Ke Hain Char Din", "artist": "Sonu Nigam, Sunidhi Chauhan", "img": "happy4.jpg", "file": "happy4.mp3"},
        {"name": "Supreme", "artist": "Shubh", "img": "happy5.jpg", "file": "happy5.mp3"},
        {"name": "Be Mine", "artist": "Shubh", "img": "happy6.jpg", "file": "happy6.mp3"},
    ],
    "Sad": [
        {"name": "Kabhi Sham Dhale", "artist": "Mohammad Faiz", "img": "sad1.jpg", "file": "sad1.mp3"},
        {"name": "Nira Ishq", "artist": "Guri", "img": "sad2.jpg", "file": "sad2.mp3"},
        {"name": "Pal Pal Dil Ke Paas", "artist": "Arijit Singh", "img": "sad3.jpg", "file": "sad3.mp3"},
        {"name": "Pal", "artist": "Arijit Singh, Javed Mohsin", "img": "sad4.jpg", "file": "sad4.mp3"},
        {"name": "Tum Hi Ho", "artist": "Mithoon, Arijit Singh", "img": "sad5.jpg", "file": "sad5.mp3"},
        {"name": "Tera Hone Laga Hoon", "artist": "Atif Aslam, Alisha Chinai", "img": "sad6.jpg", "file": "sad6.mp3"},
    ],
    "Motivational": [
        {"name": "52 Bars", "artist": "Karan Aujla", "img": "mot1.jpg", "file": "mot1.mp3"},
        {"name": "Father Saab", "artist": "Khasa Aala Chahar", "img": "mot2.jpg", "file": "mot2.mp3"},
        {"name": "Game", "artist": "Shooter Kahlon, Sidhu Moosewala", "img": "mot3.jpg", "file": "mot3.mp3"},
        {"name": "Get Ready To Fight", "artist": "Benny Dayal", "img": "mot4.jpg", "file": "mot4.mp3"},
        {"name": "Winning Speech", "artist": "Karan Aujla", "img": "mot5.jpg", "file": "mot5.mp3"},
        {"name": "Aarambh Hai Prachand", "artist": "K.K Menon, Abhimannyu Singh", "img": "mot6.jpg", "file": "mot6.mp3"},
    ],
    "Party": [
        {"name": "Abhi Toh Party Shuru Hui Hai", "artist": "Badshah, Aastha Gill", "img": "party1.jpg", "file": "party1.mp3"},
        {"name": "5 Taara", "artist": "Diljit Dosanjh, Jatinder Shah", "img": "party2.jpg", "file": "party2.mp3"},
        {"name": "Daaru Party", "artist": "Millind Gaba", "img": "party3.jpg", "file": "party3.mp3"},
        {"name": "Party All Night", "artist": "Yo Yo Honey Singh", "img": "party4.jpg", "file": "party4.mp3"},
        {"name": "Kala Chashma", "artist": "Prem Hardeep, Badshah, Neha Kakkar", "img": "party5.jpg", "file": "party5.mp3"},
        {"name": "Tujhe Aksa Beach Ghuma Doon", "artist": "Wajid, Amrita Kak", "img": "party6.jpg", "file": "party6.mp3"},
    ],
}


def _local_image_url(song):
    image_path = os.path.join("static", "image", song.get("img", ""))
    if not os.path.exists(image_path):
        # No custom placeholder shipped — reuse the background artwork so
        # a missing cover never shows a broken image.
        return url_for("static", filename="img/music-bg.svg")
    return url_for("static", filename=f"image/{song['img']}")


def build_fallback_payload(song):
    """Only used if Spotify search couldn't return anything at all."""
    song = dict(song)
    song["spotify_url"] = None
    song["embed_url"] = None
    song["image_url"] = _local_image_url(song)
    song["audio_url"] = url_for("serve_audio", filename=song["file"])
    return song


def get_songs_for_mood(mood, count=6):
    """
    Primary path: live Deezer/iTunes search pool (effectively unlimited
    catalog, refreshed periodically — see music_client.get_mood_songs).
    Fallback path: the small built-in list, only if both providers are
    unreachable (e.g. no internet).
    """
    songs = music_client.get_mood_songs(mood, count=count)
    if songs:
        return songs

    logger.warning(f"Music pool empty for '{mood}' — using local fallback list.")
    picks = random.sample(
        FALLBACK_SONG_DB[mood], min(count, len(FALLBACK_SONG_DB[mood]))
    )
    return [build_fallback_payload(s) for s in picks]


def _optional_float(form, key):
    raw = (form.get(key) or "").strip()
    if not raw:
        return None
    return float(raw)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Multi-modal signals — each is optional; whichever the user
            # actually provided (webcam face read, typed text, spoken/
            # transcribed voice) gets fused into one mood. See fusion.py.
            face_emotion = (request.form.get("face_emotion") or "").strip() or None
            face_confidence = _optional_float(request.form, "face_confidence")
            text_input = (request.form.get("text_mood") or "").strip() or None
            voice_input = (request.form.get("voice_mood") or "").strip() or None

            # Manual sliders — the original input mode, now used as the
            # fallback path when no face/text/voice signal is usable.
            energy = _optional_float(request.form, "energy")
            dance = _optional_float(request.form, "danceability")
            valence = _optional_float(request.form, "valence")
            for label, val in (("energy", energy), ("danceability", dance), ("valence", valence)):
                if val is not None and not (1 <= val <= 10):
                    return render_template(
                        "index.html", mood=None,
                        error="Manual sliders must be between 1 and 10.",
                    )

            mood, breakdown, modalities = fusion.detect_mood(
                face_emotion=face_emotion, face_confidence=face_confidence,
                text=text_input, voice_text=voice_input,
                energy=energy, danceability=dance, valence=valence,
                model=model, label_encoder=le,
            )

            if not mood:
                return render_template(
                    "index.html", mood=None,
                    error="Couldn't read a mood from that — let the camera see your "
                          "face, type a sentence, use the mic, or set the manual sliders.",
                )

            songs = get_songs_for_mood(mood, count=6)
            database.save_mood_event(mood, modalities, breakdown, songs)

            return render_template(
                "index.html", mood=mood, songs=songs,
                breakdown=breakdown, modalities=modalities,
            )
        except ValueError:
            return render_template(
                "index.html", mood=None, error="Please enter valid numbers"
            )
        except Exception as e:
            logger.error(f"Error occurred: {str(e)}")
            return render_template(
                "index.html", mood=None,
                error="Something went wrong on our end. Please try again.",
            )

    return render_template("index.html", mood=None)


@app.route("/log-play", methods=["POST"])
def log_play():
    """
    Called from the front end when a song actually starts playing (Spotify
    embed playback event, or the local <audio> fallback firing 'play') —
    not merely when it's shown as a recommendation. Powers 'last 3 listened'.
    """
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    artist = (data.get("artist") or "").strip()
    if not name or not artist:
        return jsonify({"ok": False, "error": "name and artist are required"}), 400

    database.log_play(
        name=name,
        artist=artist,
        mood=data.get("mood"),
        image_url=data.get("image_url"),
        spotify_url=data.get("spotify_url"),
    )
    return jsonify({"ok": True})


@app.route("/history")
def history():
    entries = database.get_history(limit=50)
    mood_counts = database.get_mood_counts()
    recent_plays = database.get_recent_plays(limit=3)
    return render_template(
        "history.html", entries=entries, mood_counts=mood_counts, recent_plays=recent_plays
    )


@app.route("/history/clear", methods=["POST"])
def clear_history():
    database.clear_history()
    return render_template(
        "history.html", entries=[], mood_counts={}, recent_plays=[], cleared=True
    )


@app.route("/static/<path:filename>")
def serve_static(filename):
    try:
        return send_from_directory("static", filename)
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {str(e)}")
        return str(e), 404


@app.route("/audio/<path:filename>")
def serve_audio(filename):
    """Local fallback audio, used only if a live search couldn't be reached."""
    try:
        audio_path = os.path.join("static", "audio", filename)
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return "Audio file not found", 404
        return send_file(audio_path, mimetype="audio/mpeg", as_attachment=False)
    except Exception as e:
        logger.error(f"Error serving audio file {filename}: {str(e)}")
        return str(e), 500


if __name__ == "__main__":
    app.run(debug=DEBUG, use_reloader=False, port=int(os.environ.get("PORT", 6767)))
