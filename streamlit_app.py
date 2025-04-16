"""
streamlit_app.py (Modified - No Streaming RMSE)

Streamlit front-end for the Machine Learning in Reserving - Diagnostic App.
- Collects hyperparameters and run information from the user.
- Downloads (or loads from cache) the dataset, providing cleaner status updates.
- Starts MLflow logging.
- Calls run_diagnostics() from _DiagnosticCode2.py within an st.status container.
- Clears previous output figures before saving new ones.
- Displays the formatted evaluation metric (MSE) and diagnostic plots, with specific handling for full-width plots.
"""

import streamlit as st
import mlflow
import pandas as pd
import os
import traceback
import time
import numpy as np # Import numpy for checking numeric types
import shutil # Import shutil for directory removal

import subprocess
import threading
import socket
import psutil

# Add logging with minimal changes
import logging
from datetime import datetime
log_filename = f"app_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the path is correct for importing local modules
import sys

# Corrected import to match user's file name
from _DiagnosticCode2 import run_diagnostics

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Function to check if MLflow UI is already running
def is_mlflow_ui_running():
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if process.info['cmdline'] and len(process.info['cmdline']) > 1:
            cmdline = ' '.join(process.info['cmdline'])
            if 'mlflow ui' in cmdline:
                return True
    
    # Also check if port 5000 is in use
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 5000))
    sock.close()
    return result == 0

# Function to start MLflow UI in a separate thread
def start_mlflow_ui():
    # Set the tracking URI first to ensure MLflow uses the right directory
    mlruns_dir = os.path.abspath("./mlruns")
    if not os.path.exists(mlruns_dir):
        os.makedirs(mlruns_dir)
    
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")
    
    # Start MLflow UI in a subprocess
    # CHANGED: Added --host 0.0.0.0 to bind to all interfaces
    subprocess.Popen(["mlflow", "ui", "--host", "0.0.0.0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # CHANGED: Updated log message to reflect all interfaces
    logging.info("MLflow UI started on all interfaces (0.0.0.0:5000)")
    print("MLflow UI started on all interfaces (0.0.0.0:5000)")

# Check if MLflow UI is running, start it if not
if not is_mlflow_ui_running():
    try:
        start_mlflow_ui()
        # Wait a moment for the server to start
        time.sleep(2)
        logging.info("MLflow UI started successfully")
        print("MLflow UI started successfully")
    except Exception as e:
        logging.error(f"Failed to start MLflow UI: {e}")
        print(f"Failed to start MLflow UI: {e}")
else:
    logging.info("MLflow UI is already running")
    print("MLflow UI is already running")

# --- Streamlit Page Setup ---
st.set_page_config(layout="wide")
st.title("Machine Learning in Reserving - Diagnostic App")

# CHANGED: Dynamic hostname detection for MLflow link
# Get server hostname/IP for MLflow link
hostname = socket.gethostname()
try:
    # Try to get the IP address
    ip_address = socket.gethostbyname(hostname)
    # For localhost development, use localhost
    if ip_address.startswith("127."):
        mlflow_url = "http://localhost:5000"
    else:
        mlflow_url = f"http://{ip_address}:5000"
except:
    # Fallback to using the same host as Streamlit
    mlflow_url = "http://" + socket.getfqdn() + ":5000"

# Add a clickable link to the MLflow UI at the top of your app
# CHANGED: Use dynamic mlflow_url instead of hardcoded localhost
st.markdown(f"""
### [📊 Open MLflow Dashboard]({mlflow_url})
Click the link above to open the MLflow tracking dashboard in a new tab.
""", unsafe_allow_html=True)

# --- User Input: Hyperparameters & Run Name ---
st.sidebar.header("Model Configuration")
default_run_name = f"reserving_diag_{time.strftime('%Y%m%d_%H%M%S')}"
user_name = st.sidebar.text_input("Please Enter Your Name", help="Enter your name for MLflow tracking.")
run_name = st.sidebar.text_input("MLflow Run Name", value=default_run_name)
st.sidebar.subheader("Hyperparameters")
nn_iter = st.sidebar.number_input("Number of Epochs (nn_iter)", min_value=1, max_value=10000, value=1000, step=100)
max_lr = st.sidebar.number_input("Max Learning Rate", min_value=0.00001, max_value=1.0, value=0.001, step=0.0001, format="%.4f")
init_bias = st.sidebar.number_input("Initial Bias (Log Scale)", value=12.0, format="%.2f", help="Initial bias for the output layer. Try log(mean(target)) if unsure.")
n_hidden = st.sidebar.number_input("Number of Hidden Nodes", min_value=4, max_value=1024, value=20, step=4)
batchnorm = st.sidebar.checkbox("Use BatchNorm", value=True)
dropout_rate = st.sidebar.slider("Dropout Rate", min_value=0.0, max_value=0.75, value=0.1, step=0.05, help="Fraction of neurons to randomly drop during training.")


# --- Cached Data Loading with Refined Status ---
@st.cache_data
def load_and_cache_data():
    """Loads dataset, handling download and caching. Returns df or None."""
    data_dir = "data"
    data_path = os.path.join(data_dir, "datwTestTrainSplit.csv")
    os.makedirs(data_dir, exist_ok=True)
    status_placeholder = st.empty()
    if not os.path.exists(data_path):
        status_placeholder.info(f"Downloading dataset to {data_path}...")
        url = "https://raw.githubusercontent.com/MLRWP/mlrwp-book/main/Research/datwTestTrainSplit.csv"
        try:
            df = pd.read_csv(url)
            df.to_csv(data_path, index=False)
        except Exception as e:
            status_placeholder.error(f"Failed to download dataset. Check URL/connection. Error: {e}")
            return None
    else:
        status_placeholder.info(f"Loading cached dataset from {data_path}...")
    try:
        df = pd.read_csv(data_path)
        required_cols = ['claim_size', 'train_ind', 'occurrence_time', 'development_period']
        if not all(col in df.columns for col in required_cols):
             status_placeholder.error(f"Loaded data missing required columns: {required_cols}")
             return None
        formatted_rows = f"{len(df):,}"
        status_placeholder.success(f"Dataset loaded ({formatted_rows} rows).")
        return df
    except Exception as e:
        status_placeholder.error(f"Failed to load dataset from {data_path}. Error: {e}")
        return None

# Load data and display status
data = load_and_cache_data()

# Initialize session state
if 'results' not in st.session_state: # Stores the final results dict
    st.session_state.results = None
if 'current_run_figures' not in st.session_state: # Stores [(name, fig)] from last successful run
    st.session_state.current_run_figures = []
if 'selected_figures' not in st.session_state: # Stores list of names of figures selected by user
    st.session_state.selected_figures = []

# --- Plot Selection (Revised Logic for Change #4) ---
st.sidebar.subheader("Diagnostic Plots")
figure_options = []
# Get figure names from the stored figures of the last successful run
if st.session_state.current_run_figures:
     figure_options = [name for name, fig in st.session_state.current_run_figures]

# Initialize selection if necessary (e.g., after first run)
if not figure_options:
     st.session_state.selected_figures = []
elif not st.session_state.selected_figures and figure_options:
    st.session_state.selected_figures = figure_options

# Let the multiselect manage the state directly
st.session_state.selected_figures = st.sidebar.multiselect(
    "Select figures to display:",
    options=figure_options,
    default=st.session_state.selected_figures, # Use the current state as default
    key="figure_multiselect"
)

if not figure_options and data is not None:
    st.sidebar.info("Run the model to generate and select plots.")
elif data is None:
    st.sidebar.warning("Load data successfully to enable model run.")

# --- Run Model Button and Status Container ---
if st.button("Train Model and Generate Diagnostics", disabled=(data is None), type="primary"):
    if data is not None:
        with st.status("Starting diagnostic run...", expanded=True) as status:
            try:
                # MLflow Setup
                mlruns_dir = os.path.abspath("./mlruns")
                if not os.path.exists(mlruns_dir):
                     os.makedirs(mlruns_dir)
                mlflow.set_tracking_uri(f"file:{mlruns_dir}")
                experiment_name = "MLRWP_Reserving_Diagnostics"
                mlflow.set_experiment(experiment_name)
                status.write(f"Starting MLflow run: '{run_name}' under experiment: '{experiment_name}'")
                logging.info(f"Starting MLflow run: '{run_name}' under experiment: '{experiment_name}'")

                with mlflow.start_run(run_name=run_name) as run:
                    run_id = run.info.run_id
                    status.write(f"MLflow Run ID: {run_id}")
                    logging.info(f"MLflow Run ID: {run_id}")
                    # (MLflow UI link generation - improved)
                    try:
                        client = mlflow.tracking.MlflowClient()
                        experiment = client.get_experiment_by_name(experiment_name)
                        exp_id = experiment.experiment_id
                        # CHANGED: Use the same mlflow_url from above instead of trying to detect again
                        mlflow_ui_url = f"{mlflow_url}/#/experiments/{exp_id}/runs/{run_id}"
                        status.markdown(f"[View MLflow Run]({mlflow_ui_url})", unsafe_allow_html=True)
                    except Exception:
                        status.warning("Could not generate MLflow UI link. Run `mlflow ui` manually.")
                        logging.warning("Could not generate MLflow UI link.")

                    mlflow.log_params({
                        "nn_iter": nn_iter, "max_lr": max_lr, "init_bias": init_bias,
                        "n_hidden": n_hidden, "batchnorm": batchnorm, "dropout_rate": dropout_rate,
                        "run_name": run_name
                    })
                    mlflow.log_param("data_rows", data.shape[0])
                    mlflow.log_param("training_rows", data[data['train_ind'] == 1].shape[0])

                    status.write("Training model and generating diagnostics...")
                    logging.info("Training model and generating diagnostics...")
                    start_time = time.time()

                    # Run Diagnostics call (No callback needed now)
                    results = run_diagnostics(
                        dat=data.copy(), nn_iter=nn_iter, max_lr=max_lr, init_bias=init_bias,
                        n_hidden=n_hidden, batchnorm=batchnorm, dropout=dropout_rate, run_name=run_name
                        # Removed progress_callback argument
                    )
                    duration = time.time() - start_time
                    status.write(f"Run diagnostics processing complete! (Duration: {duration:.2f}s)")
                    logging.info(f"Run diagnostics processing complete! (Duration: {duration:.2f}s)")

                    # Store results and figures in session state
                    st.session_state.results = results # Store final results dict
                    st.session_state.current_run_figures = results.get("figures", []) # Store [(name, fig)] list

                    # Update selected figures: default to selecting all newly generated figures (Change #4 Fix)
                    new_figure_names = [name for name, fig in st.session_state.current_run_figures]
                    st.session_state.selected_figures = new_figure_names

                    # --- Clear output directory and save figures (Change #1) ---
                    output_dir = os.path.abspath("outputs/figures")
                    status.write(f"Preparing output directory: {output_dir}")
                    if os.path.exists(output_dir):
                        try:
                            shutil.rmtree(output_dir)
                            status.write("Cleared previous output figures.")
                        except Exception as e:
                             status.warning(f"Could not clear output directory '{output_dir}': {e}")
                             logging.warning(f"Could not clear output directory '{output_dir}': {e}")
                    try:
                        os.makedirs(output_dir, exist_ok=True) # Recreate it
                    except Exception as e:
                        status.error(f"Could not create output directory '{output_dir}': {e}")
                        logging.error(f"Could not create output directory '{output_dir}': {e}")

                    # Log metrics/artifacts
                    mlflow.log_metric("mse_train", results.get("mse", float('nan')))
                    mlflow.log_metric("run_duration_sec", duration)

                    logged_plots_count = 0
                    if st.session_state.current_run_figures:
                         status.write(f"Logging {len(st.session_state.current_run_figures)} plots to MLflow artifacts...")
                         for name, fig in st.session_state.current_run_figures:
                             safe_name = "".join(c if c.isalnum() else "_" for c in name).strip("_")
                             path = os.path.join(output_dir, f"{safe_name}_{run_id[:8]}.png")
                             try:
                                 fig.savefig(path, bbox_inches='tight')
                                 mlflow.log_artifact(path, artifact_path="figures")
                                 logged_plots_count += 1
                             except Exception as e:
                                 status.warning(f"Could not save/log figure '{name}': {e}")
                                 logging.warning(f"Could not save/log figure '{name}': {e}")
                         status.write(f"Logged {logged_plots_count} plots.")
                         logging.info(f"Logged {logged_plots_count} plots.")
                    else:
                         status.warning("No figures were generated by the diagnostics run.")
                         logging.warning("No figures were generated by the diagnostics run.")

                    status.update(label="Diagnostic run completed successfully!", state="complete", expanded=False)
                    logging.info("Diagnostic run completed successfully!")
                    st.rerun() # Rerun to update the main display area

            except Exception as e:
                # Ensure MLflow run ends if it started
                try:
                    mlflow.set_tag("run_status", "FAILED")
                    mlflow.set_tag("error_message", traceback.format_exc())
                    mlflow.end_run(status="FAILED")
                except Exception as mlflow_e:
                     print(f"Error trying to fail MLflow run: {mlflow_e}") # Log to console
                     logging.error(f"Error trying to fail MLflow run: {mlflow_e}")

                status.update(label="Diagnostic run failed!", state="error", expanded=True)
                status.error("An error occurred during the diagnostic run:")
                status.exception(e)
                logging.error(f"Diagnostic run failed: {str(e)}")
                logging.error(traceback.format_exc())
                # Clear potentially partial results and figures
                st.session_state.results = None
                st.session_state.current_run_figures = []
                st.session_state.selected_figures = []


# --- Display Results Area (MSE displayed first, plots revised) ---
st.markdown("---")
st.header("Results")

if st.session_state.results:
    # --- Display Final MSE First ---
    mse_value = st.session_state.results.get('mse', None)
    if isinstance(mse_value, (int, float, np.number)) and not np.isnan(mse_value): # Check type and NaN
         formatted_mse = f"{mse_value:,.0f}"
    else:
         formatted_mse = "N/A"
    st.metric(label="Final MSE (Training Set)", value=formatted_mse)
    st.markdown("---") # Add separator after metric

    # --- Display Plots (Revised for full-width - Change #3) ---
    st.subheader("Diagnostic Plots")
    if not st.session_state.selected_figures:
        st.info("Select plots from the sidebar to display.")
    else:
        figures_available = st.session_state.current_run_figures
        if not figures_available:
            st.warning("No figures available from the last run.")
        else:
            # Separate target plot from others
            target_plot_name = "Development and Occurrence Performance"
            target_fig_data = None
            other_figs_data = []

            for name, fig in figures_available:
                if name == target_plot_name:
                    target_fig_data = (name, fig)
                else:
                    other_figs_data.append((name, fig))

            displayed_target = False
            # Display target plot full width (if selected)
            if target_fig_data and target_fig_data[0] in st.session_state.selected_figures:
                st.write(f"#### {target_fig_data[0]}")
                st.pyplot(target_fig_data[1], use_container_width=True)
                st.markdown("---") # Add separator after full-width plot
                displayed_target = True

            # Display other plots in columns (if selected)
            displayed_others_count = 0
            if other_figs_data:
                cols = st.columns(2)
                col_idx = 0
                for name, fig in other_figs_data:
                    if name in st.session_state.selected_figures:
                        with cols[col_idx % len(cols)]:
                            st.write(f"#### {name}")
                            st.pyplot(fig, use_container_width=True) # Use container width here too
                        displayed_others_count += 1
                        col_idx += 1

            # Check if any selected plots were actually displayed
            if not displayed_target and displayed_others_count == 0 and figure_options: # Added check for figure_options
                 st.warning("Plots are available from the last run, but none of the currently selected plots were found or displayed.")


elif not st.session_state.results and data is not None:
    st.info("Configure parameters and click 'Train Model' to see results.")
elif data is None:
    st.warning("Load data successfully before running the model.")


# Footer instructions (updated)
st.sidebar.markdown("---")
st.sidebar.markdown("To view detailed logs:")
# CHANGED: Update the footer link to use the dynamic mlflow_url
st.sidebar.markdown(f"1. **MLflow:** Click the [📊 Open MLflow Dashboard]({mlflow_url}) link at the top of this page.")