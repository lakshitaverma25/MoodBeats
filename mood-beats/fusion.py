"""
fusion.py
---------
The Mood Fusion Engine.

Combines up to three independent mood signals:
  - face   : dominant facial expression from the browser-side face-api.js
             model (webcam), mapped onto our 4 mood classes.
  - text   : typed text, scored by text_sentiment.analyze().
  - voice  : a speech-to-text transcript (captured client-side via the
             Web Speech API), scored the same way as text.

...into a single mood label, using confidence-weighted voting. Each
signal that fires casts a vote worth (base_weight * its own confidence).
The mood with the highest total wins. This is a transparent, explainable
fusion strategy (easy to defend in a viva) rather than a black-box model:
you can point at exactly why a mood was chosen.

If none of face/text/voice produce a usable signal (e.g. camera denied,
no text typed, mic not used), we fall back to the original manual
Energy / Danceability / Valence sliders and the trained RandomForest
classifier — so the app never dead-ends.
"""

import text_sentiment

# Facial expressions reported by face-api.js's faceExpressionNet, mapped to
# Mood Beats' four playlist buckets. "neutral" deliberately maps to None —
# a blank face isn't a confident vote for any mood.
FACE_TO_MOOD = {
    "happy": "Happy",
    "surprised": "Party",
    "sad": "Sad",
    "angry": "Motivational",   # channel intensity into a motivational playlist
    "fearful": "Sad",
    "disgusted": "Sad",
    "neutral": None,
}

WEIGHT_FACE = 0.5
WEIGHT_TEXT = 0.3
WEIGHT_VOICE = 0.2


def detect_mood(face_emotion=None, face_confidence=None, text=None,
                 voice_text=None, energy=None, danceability=None,
                 valence=None, model=None, label_encoder=None):
    """
    Returns (mood, breakdown, modalities):
      mood        - winning mood label, or None if nothing usable was given
      breakdown   - dict describing what each modality contributed, for
                    display in the UI and for the history log
      modalities  - short string like "face+text", "text", or "manual",
                    recording which signals actually produced the result
    """
    votes = {}
    breakdown = {}

    if face_emotion:
        mapped = FACE_TO_MOOD.get(face_emotion.strip().lower())
        conf = float(face_confidence) if face_confidence not in (None, "") else 0.7
        conf = max(0.0, min(1.0, conf))
        breakdown["face"] = {
            "emotion": face_emotion, "confidence": round(conf, 2), "mapped_mood": mapped,
        }
        if mapped:
            votes[mapped] = votes.get(mapped, 0.0) + WEIGHT_FACE * conf

    if text:
        mood, conf = text_sentiment.analyze(text)
        breakdown["text"] = {"input": text, "confidence": conf, "mapped_mood": mood}
        if mood:
            votes[mood] = votes.get(mood, 0.0) + WEIGHT_TEXT * conf

    if voice_text:
        mood, conf = text_sentiment.analyze(voice_text)
        breakdown["voice"] = {"input": voice_text, "confidence": conf, "mapped_mood": mood}
        if mood:
            votes[mood] = votes.get(mood, 0.0) + WEIGHT_VOICE * conf

    if votes:
        final_mood = max(votes, key=votes.get)
        breakdown["votes"] = {k: round(v, 3) for k, v in votes.items()}
        modalities = "+".join(
            m for m in ("face", "text", "voice")
            if breakdown.get(m, {}).get("mapped_mood")
        )
        return final_mood, breakdown, modalities or "unknown"

    # ---- Fallback: manual sliders + the original trained classifier ----
    if None not in (energy, danceability, valence) and model is not None and label_encoder is not None:
        prediction = model.predict([[energy, danceability, valence]])[0]
        final_mood = label_encoder.inverse_transform([prediction])[0]
        breakdown["manual"] = {
            "energy": energy, "danceability": danceability, "valence": valence,
        }
        return final_mood, breakdown, "manual"

    return None, breakdown, None
