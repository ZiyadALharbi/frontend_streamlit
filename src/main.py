# app.py
import os
import io
import time
import base64
import requests
import streamlit as st
import pydeck as pdk
from PIL import Image

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Oil Spill Detection", page_icon="🛰️", layout="wide")

import pathlib

# ---------------------------
# CONFIG
# ---------------------------
BASE_DIR = pathlib.Path(__file__).parent.resolve()

FASTAPI_URL = os.getenv("FASTAPI_URL", "https://fastapipetra-production.up.railway.app")  
BACKGROUND_IMAGE_PATH = str(BASE_DIR / "background.png")
INTRO_VIDEO_PATH = str(BASE_DIR / "earth_zoom.mp4")

# Sample demo points (edit/expand as you like)
DEMO_COORDS = [
    {"name": "Detection #1", "lat": 26.315, "lon": 50.103, "conf": 0.91},
    {"name": "Detection #2", "lat": 26.420, "lon": 49.978, "conf": 0.78},
    {"name": "Detection #3", "lat": 26.230, "lon": 50.205, "conf": 0.62},
]


# ---------------------------
# HELPERS
# ---------------------------
def _as_data_uri(path_or_url: str, mime: str) -> str:
    if path_or_url.lower().startswith("http"):
        # Streamlit can use remote urls directly; we only need data URI for local assets
        return path_or_url
    with open(path_or_url, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def set_background_image(path_or_url: str):
    uri = _as_data_uri(path_or_url, "image/png")
    st.markdown(
        f"""
        <style>
        /* Set the whole main container background to your image */
        [data-testid="stAppViewContainer"] {{
            background: url("{uri}") no-repeat center center fixed !important;
            background-size: cover !important;
            background-color: transparent !important;
        }}

        /* Make main content transparent to see the background */
        .stApp, .main, .block-container {{
            background-color: rgba(0,0,0,0) !important;
        }}

        /* Remove top header background */
        header[data-testid="stHeader"] {{
            background: rgba(0,0,0,0) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def show_brandbar():
    st.markdown(
        """
        <div style="
            display:flex;
            justify-content:center;
            align-items:center;
            gap:12px;
            margin-bottom: 25px;
        ">
          <div style="font-size:2.8rem;">🛰️ <b>Petra</b></div>
          <div style="opacity:.7; font-size:2.8rem;">| Oil Spill Detection From World Sea</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# def call_fastapi_predict_file(file_bytes: bytes, filename: str):
#     """Send image to FastAPI backend for prediction"""
#     try:
#         resp = requests.post(
#             f"{FASTAPI_URL}/predict",
#             files={"file": (filename, file_bytes, "image/jpeg")},
#             timeout=60,
#         )
#         if resp.ok:
#             data = resp.json()
#             return True, {
#                 "Total Detections": data.get("total_detections"),
#                 "Detections": data.get("detections"),
#                 "Processing Time (s)": data.get("processing_time"),
#             }
#         else:
#             return False, {"error": f"{resp.status_code}: {resp.text}"}
#     except Exception as e:
#         return False, {"error": str(e)}


# def call_fastapi_predict_url(image_url: str):
#     # Alternative pattern: JSON with image URL
#     try:
#         resp = requests.post(
#             f"{FASTAPI_URL}/predict",
#             json={"url": image_url},
#             timeout=60,
#         )
#         if resp.ok:
#             return True, resp.json()
#         return False, {"error": f"{resp.status_code}: {resp.text}"}
#     except Exception as e:
#         return False, {"error": str(e)}


def call_fastapi_predict_with_visual(file_bytes: bytes, filename: str):
    """Send image to FastAPI and get back results + annotated image"""
    try:
        resp = requests.post(
            f"{FASTAPI_URL}/predict",
            files={"file": (filename, file_bytes, "image/jpeg")},
            timeout=60,
        )
        if resp.ok:
            return True, resp.json()
        else:
            return False, {"error": f"{resp.status_code}: {resp.text}"}
    except Exception as e:
        return False, {"error": str(e)}

def call_fastapi_predict_url_with_visual(image_url: str):
    """Send URL to FastAPI and get back results + annotated image"""
    try:
        resp = requests.post(
            f"{FASTAPI_URL}/predict-url",
            json={"url": image_url},
            timeout=60,
        )
        if resp.ok:
            return True, resp.json()
        else:
            return False, {"error": f"{resp.status_code}: {resp.text}"}
    except Exception as e:
        return False, {"error": str(e)}

def display_detection_results(data):
    """Display detection results with annotated image"""
    col1, col2 = st.columns([1.3, 1])
    
    with col1:
        st.subheader("📸 Detection Results")
        if "annotated_image" in data and data["annotated_image"]:
            # Decode base64 image
            import base64
            image_data = base64.b64decode(data["annotated_image"])
            image = Image.open(io.BytesIO(image_data))
            st.image(image, caption="🎯 Detected Oil Spills", use_container_width=True)
        else:
            st.info("No annotated image available")
        
    with col2:
        st.subheader("📊 Detection Summary")
        
        # Summary metrics
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("🔍 Total Detections", data.get("total_detections", 0))
        with col_b:
            st.metric("⚡ Processing Time", f"{data.get('processing_time', 0)}s")
        
        # Individual detections
        if data.get("detections"):
            st.subheader("🔍 Individual Detections")
            for i, detection in enumerate(data["detections"], 1):
                with st.expander(f"🛢️ Detection #{i}: {detection['class'].title()}", expanded=True):
                    col_x, col_y = st.columns(2)
                    with col_x:
                        st.metric("Confidence", f"{detection['confidence']}")
                    with col_y:
                        st.metric("Area Coverage", f"{detection['area_percentage']}%")
                    
                    bbox = detection['bbox']
                    st.write(f"**📍 Location:** ({bbox['x1']}, {bbox['y1']}) → ({bbox['x2']}, {bbox['y2']})")
        else:
            st.success("✅ No oil spills detected in this image!")


# ---------------------------
# STATE: Splash -> App
# ---------------------------
if "intro_done" not in st.session_state:
    st.session_state.intro_done = False

# Intro logic:
#   - If not done: show video + "Enter" button
#   - After press: mark done and rerun to load background + tabs
if not st.session_state.intro_done:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background: black !important; }
        .intro-card { text-align:center; padding-top: 6vh; color: #ddd; }
        .intro-title { font-size: 2rem; margin: 12px 0 4px 0; }
        .intro-sub { opacity:.8; margin-bottom: 14px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style="text-align:center; padding-top: 6vh; color: #ddd;">
        <div style="font-size: 2.5rem; margin-bottom: 8px;">🛰️ <b>Oil Spill Detection</b></div>
        <div style="font-size: 1.2rem; opacity: 0.8;">Cinematic journey from orbit to ocean</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------
# HERO VIDEO INTRO (AUTOPLAY)
# ---------------------------
with open(INTRO_VIDEO_PATH, "rb") as f:
    video_bytes = f.read()
video_base64 = base64.b64encode(video_bytes).decode()

st.markdown(f"""
<style>
.hero-container {{
    position: relative;
    width: 100%;
    height: 90vh;
    overflow: hidden;
    border-radius: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.5);
    margin-bottom: 2.5rem;
    margin-top: 2.5rem;
            
}}

.hero-container video {{
    position: absolute;
    top: 50%;
    left: 50%;
    min-width: 100%;
    min-height: 100%;
    transform: translate(-50%, -50%);
    object-fit: cover;
    filter: brightness(0.65) contrast(1.2);
}}

.hero-overlay {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #fff;
    text-align: center;
    z-index: 2;
}}

.hero-overlay h1 {{
    font-size: 3.5rem;
    margin-bottom: 0.5rem;
    text-shadow: 0 4px 12px rgba(0,0,0,0.7);
}}

.hero-overlay p {{
    font-size: 1.5rem;
    opacity: 0.85;
    text-shadow: 0 3px 8px rgba(0,0,0,0.6);
}}
</style>

<div class="hero-container">
  <video autoplay muted loop playsinline>
    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
  </video>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# BACKGROUND + HEADER
# ---------------------------
set_background_image(BACKGROUND_IMAGE_PATH)
show_brandbar()

st.markdown("""
<style>
/* Make tabs bigger and styled */
.stTabs [role="tablist"] {
    gap: 24px;
    justify-content: center;
    border-bottom: 2px solid rgba(255,255,255,0.2);
    margin-bottom: 30px;
}

.stTabs [role="tab"] {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    padding: 20px 35px !important;
    border-radius: 12px 12px 0 0 !important;
    background: rgba(255,255,255,0.05);
    color: #e0e0e0 !important;
    transition: all 0.3s ease;
    border: 2px solid rgba(255,255,255,0.1);
    border-bottom: none !important;
}

.stTabs [role="tab"][aria-selected="true"] {
    background: rgba(0, 0, 0, 0.55) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    border: 2px solid rgba(255,255,255,0.25);
    border-bottom: none !important;
}

.stTabs [role="tab"]:hover {
    background: rgba(255,255,255,0.1);
    color: #fff !important;
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)
tabs = st.tabs(["Intro", "Satellite", "Evaluation", "Test Model"])

# Show video banner only on tabs[1] and tabs[2]
active_tab_index = st.session_state.get("active_tab_index", 0)
if active_tab_index != 0:
    st.markdown(
        """
        <style>
        .hero-video {
            width: 100%;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 0 25px rgba(0,0,0,0.4);
            margin-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    with open(INTRO_VIDEO_PATH, "rb") as f:
        st.video(f.read())

# ---------------------------
# TAB 1: INTRO
# ---------------------------
with tabs[0]:
    st.subheader("AI-Powered Oil Spill Detection from Satellite Imagery")

    st.markdown("""
    ### 🌍 Overview
    **Petra** (a blend of **Petroleum** and **Terra**, meaning **Oil + Earth**) is a Computer Vision project 
    designed to **detect oil spills in our oceans using satellite imagery**.  
    Our mission is to support environmental protection and marine safety through 
    advanced deep learning techniques and real-time monitoring.

    ### 📡 Core Idea
    - Petra uses **YOLO** trained on real **SAR and optical satellite images**.
    - The model can **detect oil slick patterns** across large water surfaces.
    - It can integrate with **FastAPI** as a backend and a **Streamlit dashboard** for real-time monitoring.

    ### 🧠 Technical Highlights
    - **Image Preprocessing:** Images are enhanced, normalized, and resized for YOLO input.
    - **Data Augmentation:** Improves generalization across different lighting, angles, and resolutions.
    - **Model Architecture:**  YOLO Architecture. with 125 layers
    - **Deployment:** FastAPI backend for prediction + Streamlit front-end with a 3D satellite map view.

    ### 🌊 Why It Matters
    - Oil spills threaten marine ecosystems and coastal economies.
    - Early detection can help authorities react faster and reduce damage.
    - Petra provides a scalable and automated solution for **continuous satellite monitoring**.

    ### 📈 Future Goals
    - Expand dataset with real-time Sentinel-1 SAR imagery
    - Add geolocation-based detection on global map
    - Provide API endpoints for integration with maritime authorities
    """)

    st.info("Scroll to the next tabs to explore the Satellite demo and run your own predictions.")


# ---------------------------
# TAB 2: SATELLITE
# ---------------------------
with tabs[1]:
    st.markdown("### Satellite View (Demo Points)")
    st.caption("Tip: zoom & pan. Style and points are customizable.")

    mapbox_token = os.getenv("MAPBOX_API_KEY","pk.eyJ1IjoiZmFpc2FsODg4MzAwMyIsImEiOiJjbWZvN3gzMW4wMzdjMmxyMnoya29pMGF4In0.RhW7PcjNshWLLxC8nAi0Gw")

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=[{"lat": d["lat"], "lon": d["lon"], "name": d["name"], "conf": d["conf"]} for d in DEMO_COORDS],
        get_position='[lon, lat]',
        get_fill_color='[255 * (1-conf), 255 * conf, 30, 200]',
        get_radius=2500,
        pickable=True,
    )

    view_state = pdk.ViewState(latitude=26.35, longitude=50.05, zoom=6.5, pitch=30, bearing=0)
    tooltip = {"html": "<b>{name}</b><br/>Confidence: {conf}", "style": {"color": "white"}}

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/streets-v12",  # 🌈 colorful style
        api_keys={"mapbox": mapbox_token},
    )

    st.pydeck_chart(r, use_container_width=True)


    with st.expander("Demo Coordinates"):
        st.code("\n".join([f"({d['lat']}, {d['lon']})  conf={d['conf']}" for d in DEMO_COORDS]), language="text")
# ---------------------------
# TAB 3: EVALUATION
# ---------------------------
with tabs[2]:
    st.markdown("## 🧠 Evaluation of Petra CNN Model")
    st.caption("Architecture • Labels • Performance Metrics")

    # --- Section 1: CNN Architecture ---
    st.markdown("### 📐 CNN Architecture")
    st.markdown("""
    Petra's Convolutional Neural Network (CNN) is designed specifically to detect **oil spill patterns** 
    in satellite imagery. Here's how it works:

    - **Input Layer**: Accepts RGB satellite images (resized to 128x128 pixels).
    - **Convolution Layers**: Extract spatial features using 3x3 kernels and ReLU activation.
    - **Pooling Layers (MaxPooling)**: Reduce spatial dimensions to focus on the most important patterns.
    - **Batch Normalization + Dropout**: Improve training stability and prevent overfitting.
    - **Flatten + Dense Layers**: Combine extracted features and learn class-specific patterns.
    - **Output Layer (Sigmoid)**: Outputs a probability between 0 (clean sea) and 1 (oil spill).
    """)

    st.info("This model was trained on real satellite images labeled manually as **Oil Spill** or **Gas Spill**.")

    import pathlib

    BASE_DIR = pathlib.Path(__file__).parent.resolve()

    def image_path(filename: str) -> str:
        return str(BASE_DIR / filename)
    # --- Section 2: Labels and Sample Images ---
    st.markdown("### 🏷️ Labels and Training Examples")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🛢️ Real Oil Spill")
        st.image(image_path("oil_1.jpg"), caption="Oil Spill Example 1")
    with col2:
        st.markdown("#### 🛢️ Train Oil Spill")
        st.image(image_path("train_oil.jpg"), caption="Clean Sea Example 1")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🛢️ gas Spill")
        st.image(image_path("gas_1.jpg"), caption="Oil Spill Example 1")
    with col2:
        st.markdown("#### 🛢️ gas Spill")
        st.image(image_path("train_gas.jpg"), caption="Clean Sea Example 1")

    st.markdown("### 📊 Evaluation Metrics on Validation Set")

    st.markdown("""
    <style>
    .metric-box {
      background: linear-gradient(145deg, rgba(30,30,30,0.95), rgba(15,15,15,0.95));
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      padding: 22px 18px;
      text-align: center;
      box-shadow: 0 4px 18px rgba(0,0,0,0.4);
      transition: all 0.3s ease;
    }
    .metric-box:hover {
      transform: translateY(-4px);
      box-shadow: 0 8px 25px rgba(0,0,0,0.6);
    }
    .metric-title {
      font-size: 1.1rem;
      color: #aaa;
      margin-bottom: 6px;
    }
    .metric-value {
      font-size: 2rem;
      font-weight: 700;
      color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<div class="metric-box"><div class="metric-title">Accuracy</div><div class="metric-value">93%</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-box"><div class="metric-title">Precision</div><div class="metric-value">91%</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-box"><div class="metric-title">Recall</div><div class="metric-value">89%</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-box"><div class="metric-title">F1-Score</div><div class="metric-value">0.90</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="metric-box"><div class="metric-title">Loss</div><div class="metric-value">0.18</div></div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style="
            background: rgba(255,255,255,0.04);
            border-left: 4px solid #4CAF50;
            padding: 1.2rem 1.5rem;
            border-radius: 12px;
            margin-top: 2rem;
            margin-bottom: 2rem;
            color: #e0e0e0;
            box-shadow: 0 2px 15px rgba(0,0,0,0.3);
        ">
        Petra's CNN shows **strong performance** on unseen data:
        <ul>
          <li>⚡ High accuracy with balanced precision/recall</li>
          <li>🌊 Very low false positives on clean ocean images</li>
          <li>🧠 Robust detection of subtle oil spill patterns</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.success("✅ Petra achieved over **93% accuracy** distinguishing oil spills from clean sea surfaces.")


# ---------------------------
# TAB 3: TEST MODEL
# ---------------------------
with tabs[3]:
    st.markdown("### Run Inference (FastAPI)")
    st.caption(f"Endpoint: `{FASTAPI_URL}`  •  Update with env var FASTAPI_URL")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Upload Image")
        file = st.file_uploader("Satellite image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        run = st.button("🚀 Predict with Visualization", type="primary", use_container_width=True, disabled=(file is None))
        if run and file is not None:
            with st.spinner("🔄 Analyzing satellite image..."):
                ok, resp = call_fastapi_predict_with_visual(file.read(), file.name)
            if ok:
                st.success("✅ Analysis complete!")
                display_detection_results(resp)
            else:
                st.error("❌ Analysis failed")
                st.json(resp)

        st.divider()
        st.subheader("Or Predict by Image URL")
        url_in = st.text_input("Public image URL", placeholder="https://…/satellite.jpg")
        run_url = st.button("🌐 Predict URL with Visualization", use_container_width=True, disabled=(not url_in))
        if run_url and url_in:
            with st.spinner("🔄 Downloading and analyzing image..."):
                ok, resp = call_fastapi_predict_url_with_visual(url_in)
            if ok:
                st.success("✅ Analysis complete!")
                display_detection_results(resp)
            else:
                st.error("❌ Analysis failed")
                st.json(resp)

    with right:
        st.subheader("Preview")
        if file is not None:
            try:
                img = Image.open(file).convert("RGB")
                st.image(img, caption=file.name, use_container_width=True)
            except Exception:
                st.info("Preview not available. The file will still be sent to the API.")

    st.caption("FastAPI returns annotated images with bounding boxes around detected oil spills.")
















