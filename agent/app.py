import streamlit as st
from pathlib import Path
from pipeline import run_pipeline, get_artifacts_by_version

# Import your pipeline functions
# from pipeline import run_pipeline, get_artifacts_by_version, record_pipeline_version

st.title("EA Refinement Dashboard")

# Sidebar inputs
ea_file = st.text_input("EA File Path", "ExpertAdvisor.mq5")
header_files = st.text_area("Header Files (comma-separated)", "header1.mqh,header2.mqh").split(",")
set_file = st.text_input("Set File Path", "settings.set")
user_prompt = st.text_area("Optional User Guidance")

# Metrics input
profit_factor = st.number_input("Profit Factor", value=1.5)
drawdown = st.number_input("Drawdown (%)", value=10.0)
win_rate = st.number_input("Win Rate (%)", value=60.0)
metrics = {"profit_factor": profit_factor, "drawdown": drawdown, "win_rate": win_rate}

log_dir = Path("logs")

# Option 1: Select from existing log files
log_files = list(log_dir.glob("*.log"))
log_choices = [f.name for f in log_files]

selected_log = st.selectbox("Choose a log file:", log_choices)

if selected_log:
    log_path = log_dir / selected_log
    log_text = log_path.read_text(encoding="utf-16")
    st.text_area("Log Preview", log_text[:1000], height=200)

    if st.button("Run Pipeline with Selected Log"):
        # Pass log_text into your pipeline
        run_pipeline(
            vectorstore=None,
            ollama_model="ollama_model",
            ea_file="ExpertAdvisor.mq5",
            header_files=["header1.mqh", "header2.mqh"],
            set_file="settings.set",
            log_text=log_text,
            metrics={"profit_factor": 1.5, "drawdown": 10, "win_rate": 60},
            ea_code="EA code here",
            user_prompt=None
        )
        st.success(f"Pipeline run complete with {selected_log}")

# Query dashboard
version_id = st.number_input("View Version ID", min_value=1, step=1)
if st.button("Get Artifacts"):
    artifacts = get_artifacts_by_version(None, version_id)
    st.subheader(f"Artifacts for Version {version_id}")
    st.text_area("Log", artifacts.get("log", {}).get("content", ""), height=200)
    st.text_area("Analysis", artifacts.get("analysis", {}).get("content", ""), height=200)
    st.text_area("Refined Code", artifacts.get("refined_code", {}).get("content", ""), height=200)
    st.text_area("Verdict", artifacts.get("verdict", {}).get("content", ""), height=200)
