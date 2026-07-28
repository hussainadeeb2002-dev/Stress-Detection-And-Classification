import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time

try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except Exception:
    BACKEND_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="ECG Stress Prediction System",
    layout="wide"
)

# =========================================================
# HEADER
# =========================================================

st.title("ECG Stress Prediction System")

st.caption(
    "Machine Learning-Based Physiological Stress Prediction from ECG Signals"
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("System Information")

st.sidebar.write("Model: PyTorch MLP")
st.sidebar.write("Dataset: WESAD")
st.sidebar.write("Signal Type: ECG")
st.sidebar.write("Sampling Rate: 700 Hz")
st.sidebar.write("Backend: FastAPI")
st.sidebar.write("Frontend: Streamlit")

st.sidebar.markdown("---")

st.sidebar.info(
    "Upload baseline and target ECG signals to perform AI-based stress analysis."
)

# =========================================================
# FILE UPLOADS
# =========================================================

st.subheader("Upload ECG Files")

use_demo = st.checkbox("Use Demo ECG Files")

upload_col1, upload_col2 = st.columns(2)

if use_demo:

        demo_type = st.selectbox(
            "Choose a Demo Sample",
            [
            "Select Demo Sample...",
            "low_stress_sample",
            "medium_stress_sample",
            "high_stress_sample"
            ]
        )

        folder_map = {
            "low_stress_sample": "samples/low_stress",
            "medium_stress_sample": "samples/medium_stress",
            "high_stress_sample": "samples/high_stress"
        }

        baseline_file = None
        target_file = None

        if demo_type != "Select Demo Sample...":

            folder = folder_map[demo_type]

            baseline_file = open(f"{folder}/baseline.csv", "rb")
            target_file = open(f"{folder}/target.csv", "rb")


else:

    with upload_col1:
        baseline_file = st.file_uploader(
            "Upload Baseline ECG CSV",
            type=["csv"]
        )

    with upload_col2:
        target_file = st.file_uploader(
            "Upload Target ECG CSV",
            type=["csv"]
        )

# =========================================================
# MAIN WORKFLOW
# =========================================================

if baseline_file and target_file:


    left_col, right_col = st.columns([1, 2])

    # =====================================================
    # ECG VISUALIZATION
    # =====================================================

    with right_col:

        baseline_df = pd.read_csv(baseline_file)

        baseline_file.seek(0)

        target_df = pd.read_csv(target_file)

        target_file.seek(0)

        st.subheader("ECG Signal Visualization")

        baseline_chart = px.line(
            baseline_df.iloc[:3000],
            title="Baseline ECG Signal"
        )

        baseline_chart.update_layout(
            xaxis_title="Samples",
            yaxis_title="Amplitude"
        )

        target_chart = px.line(
            target_df.iloc[:3000],
            title="Target ECG Signal"
        )

        target_chart.update_layout(
            xaxis_title="Samples",
            yaxis_title="Amplitude"
        )

        st.plotly_chart(
            baseline_chart,
            use_container_width=True
        )

        st.plotly_chart(
            target_chart,
            use_container_width=True
        )

        # Reset file pointers after reading
        baseline_file.seek(0)
        target_file.seek(0)

    # =====================================================
    # PREDICTION PANEL
    # =====================================================

    with left_col:

        st.subheader("Prediction Dashboard")

        if st.button("Analyze"):

            start_time = time.time()

            with st.spinner("Running ECG stress analysis..."):

                baseline_file.seek(0)
                target_file.seek(0)

                files = {
                    "baseline_file": (
                        "baseline.csv",
                        baseline_file,
                        "text/csv"
                    ),
                    "target_file": (
                        "target.csv",
                        target_file,
                        "text/csv"
                    )
                }

                try:
                    response = requests.post(BACKEND_URL,files=files,timeout=60)

                    end_time = time.time()

                    if response.status_code == 200:
                       result = response.json()
                       prediction = result["prediction"]
                       score = result["stress_score"]
                       stress_level = result["stress_level"]
                       interpretation = result["interpretation"]

                       st.markdown("---")
                       st.subheader("Analysis Result")

                    # =====================================
                    # STATUS MESSAGE
                    # =====================================

                       if score > 0.80:
                          st.error("Severe physiological stress pattern detected.")
                       elif score > 0.60:
                          st.warning("High stress response detected.")
                       else:
                          st.success("Low or moderate stress pattern detected.")

                    # =====================================
                    # METRICS
                    # =====================================

                       st.markdown("---")
                       st.metric(label="Stress Probability Score",value=f"{score:.2%}")
                       st.metric(label="Stress Severity",value=stress_level)
                       st.subheader("Interpretation")
                       st.info(interpretation)

                       processing_time = end_time - start_time
                       st.metric(label="Inference Time",value=f"{processing_time:.2f} s")

                    else:
                        st.error(f"Backend Error ({response.status_code})")
                        st.write(response.text) 

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Cannot connect to backend server. Please ensure FastAPI is running."
                    )

                except requests.exceptions.Timeout:

                    st.error(
                        "Request timed out during ECG analysis."
                    )

                except Exception as e:

                    st.error("Unexpected error occurred.")

                    st.write(str(e))