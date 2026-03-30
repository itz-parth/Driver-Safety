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
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-info {
        background-color: #4CAF50; 
        color: white; 
        padding: 1rem; 
        border-radius: 0.5rem; 
        border-left: 5px solid #2E7D32; 
        font-size: 1.1rem; 
        font-weight: 500; 
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    @keyframes pulse { 
        0%, 100% { opacity: 1; } 
        50% { opacity: 0.7; } 
    }
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #e0e0e0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0.2rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .status-normal { border-left-color: #4CAF50; }
    .status-warning { border-left-color: #FF9800; }
    .status-error { border-left-color: #F44336; }
    .video-container { 
        border: 4px solid #333; 
        border-radius: 10px; 
        overflow: hidden; 
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2000/2000621.png", width=80)
    st.header("⚙️ System Controls")
    run = st.toggle("📹 Start Camera", value=False)
    mode = st.radio("Detection Mode", ["Drowsiness", "Distraction", "Heart Health"], index=0, horizontal=True)
    st.divider()
    # Threshold sliders (only applicable to Drowsiness mode)
    ear_thresh = st.slider("Eye Closure (EAR)", 0.10, 0.30, 0.15, 0.01, help="Eye Aspect Ratio threshold for drowsiness detection")
    mar_thresh = st.slider("Yawn (MAR)", 0.30, 0.80, 0.50, 0.01, help="Mouth Aspect Ratio threshold for yawning detection")
    nod_thresh = st.slider("Head Nod Ratio", 0.30, 0.60, 0.45, 0.01, help="Head nod ratio threshold for drowsiness detection")

st.markdown("# 🚗 Real-Time Driver Safety Monitor")
st.markdown("*Advanced AI-powered monitoring for drowsiness, distraction, and heart health*")
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
                    alert_class = "alert-drowsiness" if a["type"] == "error" else "alert-warning"
                    st.markdown(f'<div class="{alert_class}">{a["text"]}</div>', unsafe_allow_html=True)
                else: 
                    st.markdown('<div class="alert-info">✅ Driver Safety System Active</div>', unsafe_allow_html=True)
            
            with metrics_placeholder.container():
                if mode == "Heart Health" and heart_prediction:
                    # Determine status colors
                    status_class = ""
                    status_color = ""
                    if heart_prediction['class'] == 'NORMAL':
                        status_class = "status-normal"
                        status_color = "#4CAF50"
                    elif heart_prediction['class'] == 'WARNING':
                        status_class = "status-warning"
                        status_color = "#FF9800"
                    else:  # EMERGENCY
                        status_class = "status-error"
                        status_color = "#F44336"
                    
                    # Heart Rate and Condition in styled containers
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f'''
                        <div class="metric-container {status_class}">
                            <div class="metric-label">Heart Rate</div>
                            <div class="metric-value" style="color: {status_color};">{int(bpm)} BPM</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'''
                        <div class="metric-container {status_class}">
                            <div class="metric-label">Condition</div>
                            <div class="metric-value" style="color: {status_color};">{heart_prediction['class']}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    st.markdown("### Real-Time ECG (Vital Signs)")
                    st.line_chart(list(st.session_state.ecg_buffer), height=200, use_container_width=True)
                    
                    st.markdown("### Risk Analysis")
                    st.bar_chart(heart_prediction['probabilities'])
                
                elif mode == "Drowsiness":
                    st.subheader("📊 Drowsiness Monitoring")
                    if results.multi_face_landmarks:
                        # Display drowsiness metrics with visual indicators
                        col1, col2, col3 = st.columns(3)
                        
                        # EAR Metric
                        ear_status = "status-error" if res["ear"] < ear_thresh else "status-normal"
                        ear_color = "#F44336" if res["ear"] < ear_thresh else "#4CAF50"
                        with col1:
                            st.markdown(f'''
                            <div class="metric-container {ear_status}">
                                <div class="metric-label">EAR (Eye Aspect Ratio)</div>
                                <div class="metric-value" style="color: {ear_color};">{res['ear']:.3f}</div>
                                <div style="font-size: 0.8rem; color: #666;">Threshold: {ear_thresh}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        # MAR Metric
                        mar_status = "status-warning" if res["mar"] > mar_thresh else "status-normal"
                        mar_color = "#FF9800" if res["mar"] > mar_thresh else "#4CAF50"
                        with col2:
                            st.markdown(f'''
                            <div class="metric-container {mar_status}">
                                <div class="metric-label">MAR (Mouth Aspect Ratio)</div>
                                <div class="metric-value" style="color: {mar_color};">{res['mar']:.3f}</div>
                                <div style="font-size: 0.8rem; color: #666;">Threshold: {mar_thresh}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        # Nod Ratio Metric
                        nod_status = "status-warning" if res["nod_ratio"] > nod_thresh else "status-normal"
                        nod_color = "#FF9800" if res["nod_ratio"] > nod_thresh else "#4CAF50"
                        with col3:
                            st.markdown(f'''
                            <div class="metric-container {nod_status}">
                                <div class="metric-label">Head Nod Ratio</div>
                                <div class="metric-value" style="color: {nod_color};">{res['nod_ratio']:.3f}</div>
                                <div style="font-size: 0.8rem; color: #666;">Threshold: {nod_thresh}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        # Additional info with better styling
                        if res['yawn_count'] > 0:
                            st.markdown(f'''
                            <div class="metric-container">
                                <div class="metric-label">Yawn Count</div>
                                <div class="metric-value" style="color: #FF9800;">{res['yawn_count']}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        # Status indicators
                        status_col1, status_col2, status_col3 = st.columns(3)
                        with status_col1:
                            if res['drowsiness_alert']:
                                st.markdown('<div class="alert-drowsiness">🚨 DROWSINESS ALERT</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="alert-info">✅ Normal Blinking</div>', unsafe_allow_html=True)
                        with status_col2:
                            if res['yawn_alert']:
                                st.markdown('<div class="alert-warning">😴 YAWNING DETECTED</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="alert-info">✅ No Yawning</div>', unsafe_allow_html=True)
                        with status_col3:
                            if res['nod_alert']:
                                st.markdown('<div class="alert-warning">😪 HEAD NODDING</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="alert-info">✅ Stable Head Position</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="alert-warning">⚠️ No face detected - Please ensure camera is visible</div>', unsafe_allow_html=True)
                
                elif mode == "Distraction":
                    st.subheader("📊 Distraction Monitoring")
                    # Display distraction metrics with visual feedback
                    if 'metrics_output' in locals() and metrics_output:
                        # Parse and display metrics in a more visual way
                        st.markdown(f'''
                        <div class="metric-container">
                            <div class="metric-label">Distraction Status</div>
                            <div class="metric-value" style="color: #FF9800; font-weight: bold;">
                                DETECTED
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                        st.text(metrics_output)
                    else:
                        st.markdown(f'''
                        <div class="metric-container status-normal">
                            <div class="metric-label">Attention Status</div>
                            <div class="metric-value" style="color: #4CAF50; font-weight: bold;">
                                FOCUSED
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                        st.write("Analyzing driver attention...")
                

                    
        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
else:
    st.info("👈 Start camera.")
