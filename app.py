# app.py
import cv2
import numpy as np
import streamlit as st
from predict import predict_emotions_bgr

st.set_page_config(page_title="Emotion Vision", page_icon="😃", layout="centered")

st.title("Emotion Vision 😃")
st.caption("Face detection + emotion classification. Powered by TensorFlow & OpenCV.")

mode = st.radio("Acquisition Mode", ["Webcam (Live)", "Upload Image"], horizontal=True)


# --------------------------
# Utility: annotate detections
# --------------------------
def annotate(frame, detections):
    for d in detections:
        x, y, w, h = d["box"]
        label = f"{d['label']} ({d['score']:.2f})"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


# --------------------------
# Upload Image Mode
# --------------------------
def run_upload_image():
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if img_file:
        file_bytes = np.frombuffer(img_file.read(), np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        det = predict_emotions_bgr(bgr)
        out = annotate(bgr.copy(), det)

        st.image(
            cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True
        )

        if det:
            st.success(", ".join([f"{d['label']} ({d['score']:.2f})" for d in det]))
        else:
            st.info("No face detected. Try another image or better lighting.")


# --------------------------
# Live Webcam Mode
# --------------------------
def run_live_video():
    st.write("Press **Start** to initiate the capture loop. Press **Stop** to terminate.")
    start = st.button("Start", key="start_btn")

    if start:
        cap = cv2.VideoCapture(0)
        stop = st.button("Stop", key="stop_btn")
        frame_placeholder = st.empty()

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                st.error("Camera not available.")
                break

            det = predict_emotions_bgr(frame)
            out = annotate(frame.copy(), det)

            frame_placeholder.image(
                cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True
            )

            # Stop button ends loop
            if stop:
                break

        cap.release()


# --------------------------
# Mode Selector
# --------------------------
if mode == "Upload Image":
    run_upload_image()
else:
    run_live_video()
