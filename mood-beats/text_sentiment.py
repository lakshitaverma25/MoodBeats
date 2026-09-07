"""
text_sentiment.py
------------------
Turns free-typed text (or a speech-to-text transcript) into one of Mood
Beats' four mood buckets: Happy, Sad, Motivational, Party.

Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) — a rule-based
sentiment model tuned for short, informal text (social-media style), which
is exactly the kind of input we expect here ("feeling great today!",
"so tired but I have to finish this").

VADER gives us valence (positive/negative), but Mood Beats needs to tell
apart two *positive* moods (Happy vs Party) and knows about a mood VADER
has no concept of at all (Motivational). A small keyword layer on top of
the compound score resolves that, without needing a heavier transformer
model (BERT) for a 4-class problem this small.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

PARTY_KEYWORDS = (
    "party", "dance", "dancing", "club", "celebrat", "turn up", "turnt",
    "festival", "weekend", "friends", "night out", "lit", "hype",
)
MOTIVATIONAL_KEYWORDS = (
    "motivat", "workout", "gym", "focus", "grind", "hustle", "push",
    "goal", "win", "success", "determined", "discipline", "grow",
    "exam", "deadline", "productive", "study",
)


def analyze(text):
    """
    Returns (mood, confidence) for a piece of text, or (None, 0.0) if the
    text is empty / too neutral to call.

    mood is one of "Happy", "Sad", "Motivational", "Party", or None.
    confidence is a 0..1 score used as this signal's weight in the fusion.
    """
    if not text or not text.strip():
        return None, 0.0

    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]
    lowered = text.lower()

    has_party_kw = any(k in lowered for k in PARTY_KEYWORDS)
    has_motiv_kw = any(k in lowered for k in MOTIVATIONAL_KEYWORDS)

    if compound >= 0.35:
        mood = "Party" if has_party_kw else "Happy"
        confidence = min(1.0, abs(compound) + (0.15 if has_party_kw else 0.0))
    elif compound <= -0.3:
        mood = "Sad"
        confidence = min(1.0, abs(compound))
    elif has_motiv_kw:
        # Neutral-ish sentiment ("tired but I have to finish this") with a
        # clear motivational/goal-directed keyword still counts.
        mood = "Motivational"
        confidence = 0.55
    elif has_party_kw:
        mood = "Party"
        confidence = 0.5
    elif compound > 0.05:
        mood = "Happy"
        confidence = abs(compound)
    elif compound < -0.05:
        mood = "Sad"
        confidence = abs(compound)
    else:
        # Truly neutral text ("it is Tuesday") — don't force a guess.
        mood, confidence = None, 0.0

    return mood, round(confidence, 2)
