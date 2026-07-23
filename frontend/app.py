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
    page_title="AI ECG Stress Detection",
    layout="wide"
)

# =========================================================
# HEADER
# =========================================================

st.title("AI ECG Stress Detection System")

st.caption(
    "Physiological stress analysis using ECG signals and machine learning"
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

    baseline_path = "sample_data/baseline.csv"

    target_path = "sample_data/target.csv"

    baseline_file = open(baseline_path, "rb")

    target_file = open(target_path, "rb")

    st.success("Demo ECG files loaded successfully.")

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

    st.success("Both ECG files uploaded successfully.")

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

        st.subheader("AI Prediction Dashboard")

        if st.button("Analyze Stress Level"):

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

                       metric_col1, metric_col2 = st.columns(2)
                       with metric_col1:
                            st.metric(label="Stress Score",value=f"{score:.2%}")
                       with metric_col2:
                            st.metric(label="Stress Level",value=stress_level)

                       st.markdown("---")

                    # =====================================
                    # INTERPRETATION
                    # =====================================

                       st.subheader("AI Interpretation")
                       st.info(interpretation)

                    # =====================================
                    # PROCESSING TIME
                    # =====================================

                       processing_time = end_time - start_time
                       st.write(f"Processing Time: {processing_time:.2f} seconds")

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