# 🎵 MoodBeats

**AI/ML Multi-Modal Mood-Based Music Recommendation System**

MoodBeats is an intelligent, multi-modal web application built with Python and Flask. Instead of relying on manual searches or generic genres, MoodBeats automatically detects a user's emotional state—through facial expressions, text sentiment, or voice input—and recommends matching songs in real-time. 

---

## ✨ Features

- **Multi-Modal Mood Detection:** Seamlessly detects mood via Face (`face-api.js`), Text (`vaderSentiment`), and Voice (Web Speech API).
- **Confidence-Weighted Fusion Engine:** Combines available input signals using an intelligent voting mechanism to determine the most accurate final mood (Happy, Sad, Motivational, Party).
- **Machine Learning Fallback:** Utilizes a trained **Random Forest Classifier** to ensure a prediction path is always available if primary multi-modal signals are absent.
- **Live Music Discovery:** Integrates directly with free APIs (Deezer & iTunes) to fetch real-time track info, cover art, and playable audio previews.
- **In-Browser Playback:** Play song previews directly from the web interface without any redirects.
- **Mood Memory & Analytics:** Leverages **SQLite** to maintain your mood history, track recently played songs, and display visual analytics.

## 🛠️ Technology Stack

- **Backend:** Python 3.x, Flask, SQLite
- **Machine Learning:** Scikit-learn (Random Forest), vaderSentiment, Pandas, NumPy
- **Frontend:** HTML5, CSS3, JavaScript, Jinja2, face-api.js, Web Speech API
- **External APIs:** Deezer API, iTunes Search API
- **Tools:** `uv` (Package Management), VS Code, Git

## 📂 Project Structure

```text
mood-beats/
├── app.py                  # Main Flask application and routes
├── fusion.py               # Fusion Engine for multi-modal confidence voting
├── text_sentiment.py       # Sentiment analysis logic
├── database.py             # SQLite database setup and history logging
├── music_client.py         # API client for Deezer/iTunes music fetching
├── mood_model.pkl          # Trained Random Forest fallback model
├── label_encoder.pkl       # Encoded mood labels
├── pyproject.toml          # Project metadata and dependencies (uv)
├── .env.example            # Environment variables template
├── templates/              # HTML templates (index, history, base)
└── static/                 # CSS, JS (mood_detect.js, player.js), Images, Audio
```

## 🚀 Installation & Setup

MoodBeats uses [`uv`](https://github.com/astral-sh/uv) for fast and reliable Python package management.

**1. Clone the repository**
```bash
git clone https://github.com/lakshitaverma25/MoodBeats.git
cd MoodBeats
```

**2. Set up the environment**
Make sure you have Python 3.13 (as specified in `_python-version`).
```bash
uv venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
uv pip install -r requirements.txt
```

**3. Configure Environment Variables**
Copy the example environment file and update any necessary keys.
```bash
cp .env.example .env
```

**4. Run the Application**
```bash
python3 app.py
```
*The app will be available locally at `http://127.0.0.1:6767`.*

## 💡 How It Works

1. **Input Collection:** The web dashboard accesses the camera, microphone, or text input based on user preference.
2. **Signal Processing:** Each input type is parsed for mood indicators and assigned a confidence score.
3. **Fusion:** `fusion.py` aggregates the scores. If sufficient data isn't provided, it falls back to the Random Forest model.
4. **Music Matching:** The resolved mood maps to a class, triggering a live search for matching tracks via the `music_client.py`.
5. **Playback & Memory:** Songs are displayed for playback, and the session is logged to the local SQLite database for history tracking.

## 👨‍💻 Author

Developed by **Lakshita Verma** during a 45-Day AI/ML Industrial Training program.
