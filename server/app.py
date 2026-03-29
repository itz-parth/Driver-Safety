import time
from collections import deque

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Adjust imports based on your actual file structure
from detection import DrowsinessDetector
from distraction_model import load_distraction_models, predict_driver_behavior

# ============ HELPER FUNCTIONS ============
def clamp(val, min_val=0.0, max_val=1.0):
    """Prevents Streamlit progress bars from crashing if values exceed 1.0"""
    return max(min_val, min(val, max_val))

# ============ STREAMLIT PAGE CONFIG ============
st.set_page_config(
    page_title="Driver Safety Monitor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sleek styling and animations
st.markdown("""
    <style>
    .alert-drowsiness {
        background-color: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #c92a2a;
        font-size: 1.2rem;
        font-weight: bold;
        animation: pulse 1s infinite;
        margin-bottom: 1rem;
    }
    .alert-warning {
        background-color: #ffa94d;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #d67706;
        font-size: 1.2rem;
        font-weight: bold;
        animation: pulse 1s infinite;
        margin-bottom: 1rem;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    /* Make the video frame look like a monitor */
    .video-container {
        border: 4px solid #333;
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# ============ SIDEBAR CONTROLS ============
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2000/2000621.png", width=80)
    st.header("⚙️ System Controls")
    
    run = st.toggle("📹 Start Camera", value=False)
    mode = st.radio("Detection Mode", ["Drowsiness", "Distraction"], index=0)

    st.divider()
    
    st.subheader("📊 Sensitivity Thresholds")
    ear_thresh = st.slider("Eye Closure (EAR)", min_value=0.10, max_value=0.30, value=0.15, step=0.01)
    mar_thresh = st.slider("Yawn (MAR)", min_value=0.30, max_value=0.80, value=0.50, step=0.01)
    nod_thresh = st.slider("Head Nod Ratio", min_value=0.30, max_value=0.60, value=0.45, step=0.01)
    
    st.divider()
    st.caption("Final Year Project - Driver Safety System")

# ============ MAIN LAYOUT ============
st.title("🚗 Real-Time Driver Safety Monitor")
st.markdown("Monitor driver fatigue and behavioral distractions using Computer Vision and Deep Learning.")

# Main two-column layout (Slightly adjusted ratio for better video sizing)
col_camera, col_stats = st.columns([6, 4], gap="large")

with col_camera:
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    frame_window = st.image([])
    st.markdown('</div>', unsafe_allow_html=True)

# Create persistent containers for stats to prevent UI jumping
with col_stats:
    alerts_placeholder = st.empty()
    st.divider()
    metrics_placeholder = st.empty()

# ============ MODEL SETUP ============
@st.cache_resource
def init_mediapipe():
    mp_face = mp.solutions.face_mesh
    return mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                            min_detection_confidence=0.5, min_tracking_confidence=0.5)

@st.cache_resource
def get_distraction_models():
    return load_distraction_models()

face_mesh = init_mediapipe()
detector = DrowsinessDetector()
models = get_distraction_models()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# ============ STATE MANAGEMENT ============
if "alert_state" not in st.session_state:
    st.session_state.alert_state = {"text": "", "type": "success", "expires": 0}

def get_active_alert():
    now = time.time()
    if st.session_state.alert_state.get("expires", 0) > now and st.session_state.alert_state.get("text"):
        return st.session_state.alert_state
    return {"text": "", "type": "success", "expires": 0}

def set_alert(text, alert_type, duration=4):
    st.session_state.alert_state.update({
        "text": text,
        "type": alert_type,
        "expires": time.time() + duration
    })

# ============ CAMERA STREAM ============
if run:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Camera not detected. Please check your connection.")
    else:
        last_update = 0.0
        frame_times = deque(maxlen=30)
        latest_fps = 0.0

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read from camera")
                break

            frame = cv2.resize(frame, (640, 480))
            frame_times.append(time.time())
            if len(frame_times) >= 2:
                latest_fps = len(frame_times) / (frame_times[-1] - frame_times[0])

            # ================= REAL-TIME PROCESSING =================
            if mode == "Drowsiness":
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # NOTE: You can pass ear_thresh, mar_thresh here if you update detector.py!
                        detection_results = detector.detect(face_landmarks, frame.shape)
                        
                        if detection_results["drowsiness_alert"]:
                            set_alert("🚨 DROWSINESS DETECTED: WAKE UP!", "error")
                        elif detection_results["nod_alert"]:
                            set_alert("⚠️ HEAD NODDING DETECTED", "warning")
                        elif detection_results["yawn_alert"]:
                            set_alert("🥱 FREQUENT YAWNING", "warning")

            elif mode == "Distraction":
                frame, metrics_output, behavior_label = predict_driver_behavior(
                    frame, models, face_cascade, eye_cascade
                )
                if "DISTRACTED" in behavior_label:
                    set_alert(f"⚠️ DISTRACTED: {behavior_label}", "warning")

            # ================= UI UPDATE (THROTTLED TO 10 FPS) =================
            now = time.time()
            if now - last_update >= 0.1:  # Faster UI refresh for smoother experience
                last_update = now

                # 1. Update Alerts
                with alerts_placeholder.container():
                    active_alert = get_active_alert()
                    if active_alert["text"]:
                        css_class = "alert-drowsiness" if active_alert["type"] == "error" else "alert-warning"
                        st.markdown(f'<div class="{css_class}">{active_alert["text"]}</div>', unsafe_allow_html=True)
                    else:
                        st.info("✅ Driver is alert and focused.")

                # 2. Update Metrics
                with metrics_placeholder.container():
                    st.subheader(f"📈 {mode} Metrics")
                    
                    if mode == "Drowsiness" and 'detection_results' in locals():
                        c1, c2, c3 = st.columns(3)
                        c1.metric("FPS", f"{latest_fps:.1f}")
                        c2.metric("EAR (Eye)", f"{detection_results['ear']:.3f}", 
                                  delta="Low" if detection_results['ear'] < ear_thresh else "Normal", delta_color="inverse")
                        c3.metric("MAR (Mouth)", f"{detection_results['mar']:.3f}")
                        
                        st.progress(clamp(detection_results['ear'] / 0.4), text="Eye Openness (EAR)")
                        st.progress(clamp(detection_results['mar']), text="Mouth Openness (MAR)")
                        
                        col_yawn, col_nod = st.columns(2)
                        col_yawn.metric("Yawn Count", detection_results['yawn_count'])
                        col_nod.metric("Nod Ratio", f"{detection_results['nod_ratio']:.3f}")

                    elif mode == "Distraction" and 'behavior_label' in locals():
                        c1, c2 = st.columns(2)
                        c1.metric("FPS", f"{latest_fps:.1f}")
                        c2.metric("Current State", behavior_label.replace("(DISTRACTED)", "").strip())
                        
                        st.markdown("**Raw Output Data:**")
                        st.code(metrics_output, language="text")

            # Update video frame continuously
            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
else:
    st.info("👈 Please start the camera from the sidebar to begin monitoring.")