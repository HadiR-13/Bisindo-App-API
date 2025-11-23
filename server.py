import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import threading
from collections import deque

from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Config
MODEL_PATH = "my_model.keras"
ACTIONS = ['Belajar', 'Berdiri', 'Duduk', 'Makan', 'Mandi',
           'Melihat', 'Membaca', 'Menulis', 'Minum', 'Tidur']

SEQ_LENGTH = 30
THRESHOLD = 0.5
PRED_STABILITY = 10

# ------------------------------------------------------------------------------
# Persistent server-side state
# ------------------------------------------------------------------------------
client_sequences = {}
client_predictions = {}
client_sentences = {}
client_last_seen = {}
state_lock = threading.Lock()

# ------------------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------------------
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# ------------------------------------------------------------------------------
# Persistent MediaPipe Holistic instance + lock
# ------------------------------------------------------------------------------
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
)
holistic_lock = threading.Lock()

# ------------------------------------------------------------------------------
# Keypoint extraction
# ------------------------------------------------------------------------------
def arr(lm, size, dim):
    if lm:
        return np.array([
            [p.x, p.y, p.z] + ([p.visibility] if dim == 4 else [])
            for p in lm.landmark
        ])
    return np.zeros((size, dim))

def extract_keypoints(results):
    pose = arr(results.pose_landmarks, 33, 4).flatten()
    face = arr(results.face_landmarks, 468, 3).flatten()
    left = arr(results.left_hand_landmarks, 21, 3).flatten()
    right = arr(results.right_hand_landmarks, 21, 3).flatten()
    return np.concatenate([pose, face, left, right])

# ------------------------------------------------------------------------------
# Cleanup thread removes clients inactive for 60s
# ------------------------------------------------------------------------------
def cleanup_stale_clients(ttl=60):
    while True:
        now = time.time()
        with state_lock:
            stale = [cid for cid, t in client_last_seen.items() if now - t > ttl]
            for cid in stale:
                client_sequences.pop(cid, None)
                client_predictions.pop(cid, None)
                client_sentences.pop(cid, None)
                client_last_seen.pop(cid, None)
        time.sleep(30)

threading.Thread(target=cleanup_stale_clients, daemon=True).start()

# ------------------------------------------------------------------------------
# Flask app
# ------------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": time.time()})

# ------------------------------------------------------------------------------
# Prediction endpoint
# ------------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    client_id = request.form.get("client_id") or request.remote_addr

    if "frame" not in request.files:
        return jsonify({"error": "No frame uploaded"}), 400

    file = request.files["frame"]
    img_bytes = file.read()

    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    frame = cv2.flip(frame, 1)

    # Run Holistic safely
    with holistic_lock:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

    keypoints = extract_keypoints(results)

    with state_lock:
        client_last_seen[client_id] = time.time()

        if client_id not in client_sequences:
            client_sequences[client_id] = deque(maxlen=SEQ_LENGTH)
            client_predictions[client_id] = deque(maxlen=PRED_STABILITY)
            client_sentences[client_id] = []

        seq = client_sequences[client_id]
        preds = client_predictions[client_id]
        sentence = client_sentences[client_id]

        seq.append(keypoints)

        action = None
        probs = None

        if len(seq) == SEQ_LENGTH:
            x = np.expand_dims(np.array(seq), axis=0)
            res = model.predict(x, verbose=0)[0]

            pred_idx = int(np.argmax(res))
            preds.append(pred_idx)

            if (
                len(preds) == PRED_STABILITY
                and all(p == pred_idx for p in preds)
                and res[pred_idx] > THRESHOLD
            ):
                if not sentence or ACTIONS[pred_idx] != sentence[-1]:
                    sentence.append(ACTIONS[pred_idx])
                    sentence[:] = sentence[-5:]

                action = ACTIONS[pred_idx]

            probs = {ACTIONS[i]: float(res[i]) for i in range(len(ACTIONS))}

    return jsonify({
        "action": action,
        "probabilities": probs,
        "sentence": client_sentences.get(client_id, [])
    })

# ------------------------------------------------------------------------------
# Run server
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)