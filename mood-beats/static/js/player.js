/**
 * player.js
 * ---------
 * Turns "a song was shown" into "a song was listened to", and reports it
 * to /log-play so /history can show real last-3-listened songs.
 *
 * Every song card renders a plain HTML5 <audio> element pointed at a real
 * preview URL (Deezer/iTunes, or the local fallback file) — so the
 * browser's native 'play' event is all we need.
 */

(function () {
  function logPlay(payload) {
    if (!payload.name || !payload.artist) return;
    fetch("/log-play", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).catch(() => {
      /* best-effort only — a failed log shouldn't disrupt playback */
    });
  }

  document.querySelectorAll("audio.local-audio").forEach((audio) => {
    let logged = false;
    audio.addEventListener("play", () => {
      if (logged) return;
      logged = true;
      logPlay({
        name: audio.dataset.name,
        artist: audio.dataset.artist,
        mood: audio.dataset.mood,
        image_url: audio.dataset.image,
        spotify_url: audio.dataset.spotifyUrl || null,
      });
    });
  });
})();
