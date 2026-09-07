/**
 * mood_detect.js
 * --------------
 * Client-side half of the Mood Fusion Engine:
 *
 *   1. Facial expression — face-api.js (a TensorFlow.js CNN, loaded from a
 *      CDN, weights loaded from a CDN) runs entirely in the browser against
 *      the webcam feed. We never upload video/images anywhere: only the
 *      final "happy 0.87" style label is sent to the server in a hidden
 *      form field.
 *
 *   2. Voice — the browser's native Web Speech API transcribes the mic to
 *      text locally, which is then sent as plain text and scored by the
 *      exact same server-side sentiment analyzer used for typed text.
 *
 * Both are optional and fail soft: if the camera/mic is denied, missing,
 * or the models can't load (e.g. offline), the rest of the form (typed
 * text, manual sliders) still works.
 */

(function () {
  const FACE_MODELS_URL =
    "https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights";

  // ---------------- Facial expression detection ----------------
  const video = document.getElementById("cam-video");
  const camToggle = document.getElementById("cam-toggle");
  const camStatus = document.getElementById("cam-status");
  const faceReading = document.getElementById("face-reading");
  const faceEmotionField = document.getElementById("face_emotion");
  const faceConfidenceField = document.getElementById("face_confidence");

  let camStream = null;
  let detectLoop = null;
  let modelsReady = false;

  async function ensureModelsLoaded() {
    if (modelsReady || typeof faceapi === "undefined") return modelsReady;
    camStatus.textContent = "Loading model…";
    try {
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(FACE_MODELS_URL),
        faceapi.nets.faceExpressionNet.loadFromUri(FACE_MODELS_URL),
      ]);
      modelsReady = true;
      return true;
    } catch (err) {
      console.error("face-api model load failed:", err);
      camStatus.textContent = "Model failed to load — use text or voice instead";
      return false;
    }
  }

  async function startCamera() {
    if (typeof faceapi === "undefined") {
      camStatus.textContent = "Face model unavailable — use text or voice instead";
      return;
    }
    const ok = await ensureModelsLoaded();
    if (!ok) return;

    try {
      camStream = await navigator.mediaDevices.getUserMedia({ video: {}, audio: false });
      video.srcObject = camStream;
      camStatus.textContent = "Reading…";
      camToggle.textContent = "Disable camera";

      detectLoop = setInterval(async () => {
        if (!video.videoWidth) return;
        const result = await faceapi
          .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
          .withFaceExpressions();

        if (!result || !result.expressions) {
          faceReading.textContent = "No face detected";
          return;
        }
        const sorted = Object.entries(result.expressions).sort((a, b) => b[1] - a[1]);
        const [topEmotion, topScore] = sorted[0];
        faceEmotionField.value = topEmotion;
        faceConfidenceField.value = topScore.toFixed(2);
        faceReading.textContent = `${topEmotion} (${Math.round(topScore * 100)}%)`;
      }, 700);
    } catch (err) {
      console.error("Camera access failed:", err);
      camStatus.textContent = "Camera permission denied";
    }
  }

  function stopCamera() {
    if (detectLoop) clearInterval(detectLoop);
    if (camStream) camStream.getTracks().forEach((t) => t.stop());
    camStream = null;
    video.srcObject = null;
    camStatus.textContent = "Camera off";
    camToggle.textContent = "Enable camera";
    faceReading.textContent = "No reading yet";
    faceEmotionField.value = "";
    faceConfidenceField.value = "";
  }

  if (camToggle) {
    camToggle.addEventListener("click", () => {
      if (camStream) stopCamera();
      else startCamera();
    });
  }

  // ---------------- Voice input (Web Speech API) ----------------
  const micToggle = document.getElementById("mic-toggle");
  const voiceReading = document.getElementById("voice-reading");
  const voiceField = document.getElementById("voice_mood");

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognizer = null;
  let listening = false;

  if (micToggle) {
    if (!SpeechRecognition) {
      micToggle.disabled = true;
      voiceReading.textContent = "Not supported in this browser";
    } else {
      recognizer = new SpeechRecognition();
      recognizer.continuous = false;
      recognizer.interimResults = false;
      recognizer.lang = "en-US";

      recognizer.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map((r) => r[0].transcript)
          .join(" ")
          .trim();
        voiceField.value = transcript;
        voiceReading.textContent = transcript ? `"${transcript}"` : "Didn't catch that";
      };
      recognizer.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        voiceReading.textContent = "Couldn't access the microphone";
      };
      recognizer.onend = () => {
        listening = false;
        micToggle.textContent = "Start recording";
      };

      micToggle.addEventListener("click", () => {
        if (listening) {
          recognizer.stop();
          return;
        }
        voiceField.value = "";
        voiceReading.textContent = "Listening…";
        listening = true;
        micToggle.textContent = "Stop recording";
        try {
          recognizer.start();
        } catch (err) {
          console.error("Could not start recognizer:", err);
          listening = false;
          micToggle.textContent = "Start recording";
        }
      });
    }
  }

  // Release the camera if the user navigates away mid-session.
  window.addEventListener("beforeunload", stopCamera);
})();
