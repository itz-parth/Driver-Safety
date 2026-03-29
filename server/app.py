import time
from collections import deque
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from detection import DrowsinessDetector
from distraction_model import load_distraction_models, predict_driver_behavior
from heart_monitoring import load_heart_model, predict_heart_condition, generate_demo_heart_data, calculate_bpm, generate_ecg_point

st.set_page_config(page_title="Driver Safety Monitor", page_icon="🚗", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .alert-drowsiness { background-color: #ff4b4b; color: white; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #c92a2a; font-size: 1.2rem; font-weight: bold; animation: pulse 1s infinite; margin-bottom: 1rem; }
    .alert-warning { background-color: #ffa94d; color: white; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #d67706; font-size: 1.2rem; font-weight: bold; animation: pulse 1s infinite; margin-bottom: 1rem; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
    .video-container { border: 4px solid #333; border-radius: 10px; overflow: hidden; }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2000/2000621.png", width=80)
    st.header("⚙️ System Controls")
    run = st.toggle("📹 Start Camera", value=False)
    mode = st.radio("Detection Mode", ["Drowsiness", "Distraction", "Heart Health"], index=0)
    st.divider()
    ear_thresh = st.slider("Eye Closure (EAR)", 0.10, 0.30, 0.15, 0.01)
    mar_thresh = st.slider("Yawn (MAR)", 0.30, 0.80, 0.50, 0.01)
    nod_thresh = st.slider("Head Nod Ratio", 0.30, 0.60, 0.45, 0.01)

st.title("🚗 Real-Time Driver Safety Monitor")
col_camera, col_stats = st.columns([6, 4], gap="large")

with col_camera:
    frame_window = st.empty()

with col_stats:
    alerts_placeholder = st.empty()
    metrics_placeholder = st.empty()

@st.cache_resource
def get_models():
    return mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True), DrowsinessDetector(), load_distraction_models(), load_heart_model()

face_mesh, detector, distraction_models, heart_model = get_models()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

if "alert_state" not in st.session_state:
    st.session_state.alert_state = {"text": "", "type": "success", "expires": 0}

if "ecg_buffer" not in st.session_state:
    st.session_state.ecg_buffer = deque([0.0] * 100, maxlen=100)
    st.session_state.ecg_step = 0.0

if "heart_display_class" not in st.session_state:
    st.session_state.heart_display_class = "NORMAL"
    st.session_state.heart_candidate_class = "NORMAL"
    st.session_state.heart_candidate_count = 0

def set_alert(text, alert_type, duration=4):
    st.session_state.alert_state.update({"text": text, "type": alert_type, "expires": time.time() + duration})

if run:
    cap = cv2.VideoCapture(0)
    heart_prediction, demo_rr = None, None
    last_update = 0.0
    last_analysis = 0.0
    bpm = 75 # Initialize with normal BPM
    
    while run:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (640, 480))
        
        now = time.time()
        
        if mode == "Drowsiness":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                for lm in results.multi_face_landmarks:
                    res = detector.detect(lm, frame.shape)
                    if res["drowsiness_alert"]: set_alert("🚨 DROWSINESS DETECTED", "error")
                    elif res["nod_alert"]: set_alert("⚠️ HEAD NODDING", "warning")
        elif mode == "Distraction":
            frame, metrics_output, behavior_label = predict_driver_behavior(frame, distraction_models, face_cascade, eye_cascade)
            if "DISTRACTED" in behavior_label: set_alert("⚠️ DISTRACTED", "warning")
        elif mode == "Heart Health":
            # Periodic Heart Risk Analysis (every 0.8s to avoid flickering and heavy computation)
            if now - last_analysis >= 0.8:
                last_analysis = now
                demo_rr = generate_demo_heart_data()
                raw_prediction = predict_heart_condition(heart_model, demo_rr)
                raw_class = raw_prediction["class"]

                # Debounce class transitions so one noisy prediction does not
                # immediately flip the UI state.
                if raw_class == st.session_state.heart_display_class:
                    st.session_state.heart_candidate_class = raw_class
                    st.session_state.heart_candidate_count = 0
                else:
                    if raw_class == st.session_state.heart_candidate_class:
                        st.session_state.heart_candidate_count += 1
                    else:
                        st.session_state.heart_candidate_class = raw_class
                        st.session_state.heart_candidate_count = 1

                    required_count = 2 if raw_class == "WARNING" else 3 if raw_class == "EMERGENCY" else 1
                    if st.session_state.heart_candidate_count >= required_count:
                        st.session_state.heart_display_class = raw_class
                        st.session_state.heart_candidate_count = 0

                heart_prediction = dict(raw_prediction)
                heart_prediction["class"] = st.session_state.heart_display_class
                bpm = calculate_bpm(demo_rr)
                
                if heart_prediction['trigger_sos']: set_alert("🚨 MEDICAL EMERGENCY", "error")
                elif heart_prediction['class'] == 'WARNING': set_alert("⚠️ HEART WARNING", "warning")
            
            # Continuous smooth ECG streaming (independent of analysis rate)
            # Advance step based on BPM (faster BPM = faster cycle)
            st.session_state.ecg_step = (st.session_state.ecg_step + (bpm / 1200.0)) % 1.0 # Slower scroll
            new_point = generate_ecg_point(bpm, st.session_state.ecg_step)
            st.session_state.ecg_buffer.append(new_point)

        # UI Refresh Throttled
        if now - last_update >= 0.1: # 10 FPS UI is smoother but not too heavy
            last_update = now
            with alerts_placeholder.container():
                a = st.session_state.alert_state
                if a["expires"] > now: 
                    st.markdown(f'<div class="alert-{"drowsiness" if a["type"]=="error" else "warning"}">{a["text"]}</div>', unsafe_allow_html=True)
                else: 
                    st.info("✅ Driver Safety System Active")
            
            with metrics_placeholder.container():
                if mode == "Heart Health" and heart_prediction:
                    # Professional Heart Monitor UI
                    m1, m2 = st.columns(2)
                    m1.metric("Heart Rate", f"{int(bpm)} BPM", delta="Normal" if heart_prediction['class'] == 'NORMAL' else "Abnormal", delta_color="inverse")
                    m2.metric("Condition", heart_prediction['class'])
                    
                    st.markdown("### Real-Time ECG (Vital Signs)")
                    st.line_chart(list(st.session_state.ecg_buffer), height=200, use_container_width=True)
                    
                    st.markdown("### Risk Analysis")
                    st.bar_chart(heart_prediction['probabilities'])
                
                elif mode == "Drowsiness":
                    st.subheader("📊 Drowsiness Monitoring")
                    if results.multi_face_landmarks:
                        # Display drowsiness metrics
                        m1, m2, m3 = st.columns(3)
                        m1.metric("EAR (Eye Aspect Ratio)", f"{res['ear']:.3f}")
                        m2.metric("MAR (Mouth Aspect Ratio)", f"{res['mar']:.3f}")
                        m3.metric("Nod Ratio", f"{res['nod_ratio']:.3f}")
                        
                        # Additional info
                        st.write(f"Yawn Count: {res['yawn_count']}")
                        if res['drowsiness_alert']:
                            st.error("🚨 Drowsiness Alert Active")
                        if res['yawn_alert']:
                            st.warning("😴 Yawning Detected")
                        if res['nod_alert']:
                            st.warning("😪 Head Nodding Detected")
                    else:
                        st.warning("No face detected")
                
                elif mode == "Distraction":
                    st.subheader("📊 Distraction Monitoring")
                    # Display distraction metrics if available
                    if 'metrics_output' in locals():
                        st.text(metrics_output)
                    else:
                        st.write("Processing frame...")
                

                    
        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
else:
    st.info("👈 Start camera.")
