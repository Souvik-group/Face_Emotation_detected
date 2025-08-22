# predict.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ---- Config ----
MODEL_PATH = "emotion_model.h5"
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Preload assets
_model = load_model(MODEL_PATH)
_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def _preprocess_face(gray_roi):
    face = cv2.resize(gray_roi, (48, 48)).astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)     # (48,48,1)
    face = np.expand_dims(face, axis=0)      # (1,48,48,1)
    return face

def predict_emotions_bgr(frame_bgr):
    """
    Input:  BGR frame (OpenCV)
    Output: list of dicts: [{'box':(x,y,w,h),'label':'Happy','score':0.98}, ...]
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    results = []
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        inp = _preprocess_face(roi)
        probs = _model.predict(inp, verbose=0)[0]
        idx = int(np.argmax(probs))
        results.append({
            "box": (int(x), int(y), int(w), int(h)),
            "label": EMOTIONS[idx],
            "score": float(probs[idx])
        })
    return results
