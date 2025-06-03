"""
streamlit_app.py (Optimized TensorBoard Integration)

Streamlit front-end for the Machine Learning in Reserving - Diagnostic App.
"""

import streamlit as st
import pandas as pd
import os
import traceback
import time
import numpy as np
import socket
import psutil
import subprocess
import signal
import logging
from datetime import datetime
import pytz
from streamlit.components.v1 import html
from torch.utils.tensorboard import SummaryWriter  

# Setup basic logging
log_filename = f"app_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Import diagnostics function
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from _DiagnosticCode2 import run_diagnostics

# Global variables
TENSORBOARD_PORT = 6006
TENSORBOARD_LOGDIR = "./logs"

# Function to get UK time
def get_formatted_local_time():
    local_time = datetime.now(pytz.timezone('Europe/London'))
    return local_time.strftime('%Y-%m-%d %H:%M:%S %Z')

# Improved TensorBoard process management
def find_tensorboard_process():
    """Find an existing TensorBoard process"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and 'tensorboard' in ' '.join(cmdline):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

def is_port_in_use(port):
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_tensorboard():
    """Start TensorBoard if not already running"""
    # First check if a process already exists
    tb_process = find_tensorboard_process()
    if tb_process:
        logging.info(f"TensorBoard already running (PID: {tb_process.pid})")
        return True
    
    # Then check if the port is in use
    if is_port_in_use(TENSORBOARD_PORT):
        logging.info(f"Port {TENSORBOARD_PORT} already in use, assuming TensorBoard is running")
        return True
    
    # Ensure log directory exists
    os.makedirs(TENSORBOARD_LOGDIR, exist_ok=True)
    
    # Start TensorBoard process
    try:
        # Use shell=False for better security and process management
        process = subprocess.Popen(
            ["tensorboard", "--logdir", TENSORBOARD_LOGDIR, "--port", str(TENSORBOARD_PORT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Give it a moment to start
        time.sleep(1)
        
        # Verify it's running
        if process.poll() is None and is_port_in_use(TENSORBOARD_PORT):
            logging.info(f"TensorBoard started successfully on port {TENSORBOARD_PORT}")
            return True
        else:
            logging.error("TensorBoard failed to start properly")
            return False
    except Exception as e:
        logging.error(f"Error starting TensorBoard: {e}")
        return False

# Cached data loading
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
            status_placeholder.error(f"Failed to download dataset. Error: {e}")
            return None
    else:
        status_placeholder.info(f"Loading cached dataset from {data_path}...")
    
    try:
        df = pd.read_csv(data_path)
        required_cols = ['claim_size', 'train_ind', 'occurrence_time', 'development_period']
        if not all(col in df.columns for col in required_cols):
             status_placeholder.error(f"Loaded data missing required columns")
             return None
        status_placeholder.success(f"Dataset loaded ({len(df):,} rows).")
        return df
    except Exception as e:
        status_placeholder.error(f"Failed to load dataset: {e}")
        return None

# Start TensorBoard only once at app startup
if 'tensorboard_started' not in st.session_state:
    tb_running = start_tensorboard()
    st.session_state.tensorboard_started = tb_running
    if not tb_running:
        logging.warning("TensorBoard could not be started automatically")

# Streamlit UI Setup
st.set_page_config(layout="wide", page_title="ML Reserving Diagnostic App")
st.title("Machine Learning in Reserving - Diagnostic App")
st.write(f"Local time: {get_formatted_local_time()}")

# Sidebar configuration
st.sidebar.header("Model Configuration")
default_run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
user_name = st.sidebar.text_input("Please Enter Your Name", help="Enter your name for logging.")
run_name = st.sidebar.text_input("Run Name", value=default_run_name)
st.sidebar.subheader("Hyperparameters")
nn_iter = st.sidebar.number_input("Number of Epochs (nn_iter)", min_value=1, max_value=10000, value=1000, step=100)
max_lr = st.sidebar.number_input("Max Learning Rate", min_value=0.00001, max_value=1.0, value=0.001, step=0.0001, format="%.4f")
init_bias = st.sidebar.number_input("Initial Bias (Log Scale)", value=12.0, format="%.2f")
n_hidden = st.sidebar.number_input("Number of Hidden Nodes", min_value=4, max_value=1024, value=20, step=4)
batchnorm = st.sidebar.checkbox("Use BatchNorm", value=True)
dropout_rate = st.sidebar.slider("Dropout Rate", min_value=0.0, max_value=0.75, value=0.1, step=0.05)

# Load data
data = load_and_cache_data()

# Initialize session state variables
if 'results' not in st.session_state:
    st.session_state.results = None
if 'current_run_figures' not in st.session_state:
    st.session_state.current_run_figures = []
if 'selected_figures' not in st.session_state:
    st.session_state.selected_figures = []

# Plot selection
st.sidebar.subheader("Diagnostic Plots")
figure_options = [name for name, fig in st.session_state.current_run_figures] if st.session_state.current_run_figures else []

if not figure_options:
    st.session_state.selected_figures = []
elif not st.session_state.selected_figures and figure_options:
    st.session_state.selected_figures = figure_options

st.session_state.selected_figures = st.sidebar.multiselect(
    "Select figures to display:",
    options=figure_options,
    default=st.session_state.selected_figures,
    key="figure_multiselect"
)

if not figure_options and data is not None:
    st.sidebar.info("Run the model to generate plots.")
elif data is None:
    st.sidebar.warning("Load data to enable model run.")

# Run model button
if st.button("Train Model and Generate Diagnostics", disabled=(data is None), type="primary"):
    if data is not None:
        with st.status("Starting diagnostic run...", expanded=True) as status:
            try:
                # Ensure directories exist
                os.makedirs(TENSORBOARD_LOGDIR, exist_ok=True)
                output_dir = os.path.abspath("outputs/figures")
                os.makedirs(output_dir, exist_ok=True)
                
                status.write(f"Starting training run: '{run_name}'")
                
                # Run diagnostics
                start_time = time.time()
                results = run_diagnostics(
                    dat=data.copy(), 
                    nn_iter=nn_iter, 
                    max_lr=max_lr, 
                    init_bias=init_bias,
                    n_hidden=n_hidden, 
                    batchnorm=batchnorm, 
                    dropout=dropout_rate, 
                    run_name=run_name,
                    log_dir=TENSORBOARD_LOGDIR
                )
                duration = time.time() - start_time
                
                status.write(f"Run completed in {duration:.2f}s")
                
                # Display summary
                status.write("### Model Training Summary")
                status.write(f"- **Number of Epochs**: {nn_iter}")
                status.write(f"- **Learning Rate**: {max_lr}")
                status.write(f"- **Hidden Nodes**: {n_hidden}")
                status.write(f"- **Initial Bias**: {init_bias}")
                status.write(f"- **Batch Normalization**: {'Enabled' if batchnorm else 'Disabled'}")
                status.write(f"- **Dropout Rate**: {dropout_rate}")
                status.write(f"- **Final MSE**: {results.get('mse', 'N/A'):.2f}")

                # Store results
                st.session_state.results = results
                st.session_state.current_run_figures = results.get("figures", [])
                st.session_state.selected_figures = [name for name, fig in st.session_state.current_run_figures]

                # Save figures
                if st.session_state.current_run_figures:
                    saved_count = 0
                    for name, fig in st.session_state.current_run_figures:
                        safe_name = "".join(c if c.isalnum() else "_" for c in name).strip("_")
                        fig_path = os.path.join(output_dir, f"{safe_name}.png")
                        try:
                            fig.savefig(fig_path, bbox_inches='tight')
                            saved_count += 1
                        except Exception as e:
                            status.warning(f"Could not save figure '{name}': {e}")
                    status.write(f"Saved {saved_count} plots to {output_dir}")
                
                status.update(label="Diagnostic run completed!", state="complete", expanded=False)
                st.rerun()

            except Exception as e:
                status.update(label="Diagnostic run failed!", state="error", expanded=True)
                status.error("An error occurred during the run:")
                status.exception(e)
                logging.error(f"Run failed: {str(e)}")

# Results display
st.markdown("---")
st.header("Results")

# TensorBoard and Plots tabs
tab1, tab2 = st.tabs(["TensorBoard", "Diagnostic Plots"])

with tab1:
    st.subheader("TensorBoard Visualization")
    
    # Check if TensorBoard is running
    if is_port_in_use(TENSORBOARD_PORT):
        # Create the iframe for embedded TensorBoard
        tensorboard_iframe = f"""
        <iframe 
            src="http://localhost:{TENSORBOARD_PORT}"
            width="100%" 
            height="800px" 
            style="border:none;border-radius:5px;"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
            allowfullscreen>
        </iframe>
        """
        
        # Add a direct link first
        st.markdown(f"**[Open TensorBoard in new tab](http://localhost:{TENSORBOARD_PORT})**")
        
        # Then embed the TensorBoard UI
        html(tensorboard_iframe, height=800)
    else:
        st.warning("TensorBoard is not running. Please restart the app or run TensorBoard manually.")
        st.code(f"tensorboard --logdir={TENSORBOARD_LOGDIR} --port={TENSORBOARD_PORT}", language="bash")

with tab2:
    if st.session_state.results:
        # Display MSE
        mse_value = st.session_state.results.get('mse', None)
        if isinstance(mse_value, (int, float, np.number)) and not np.isnan(mse_value):
             formatted_mse = f"{mse_value:,.0f}"
        else:
             formatted_mse = "N/A"
        st.metric(label="Final MSE (Training Set)", value=formatted_mse)
        st.markdown("---")

        # Display model parameters
        st.subheader("Model Parameters")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Number of Epochs**: {nn_iter}")
            st.write(f"**Learning Rate**: {max_lr}")
            st.write(f"**Initial Bias**: {init_bias}")
        with col2:
            st.write(f"**Hidden Nodes**: {n_hidden}")
            st.write(f"**Batch Normalization**: {'Enabled' if batchnorm else 'Disabled'}")
            st.write(f"**Dropout Rate**: {dropout_rate}")
        st.markdown("---")

        # Display plots
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
                target_fig = next((fig for name, fig in figures_available if name == target_plot_name), None)
                other_figs = [(name, fig) for name, fig in figures_available if name != target_plot_name]

                # Display target plot full width (if selected)
                if target_fig and target_plot_name in st.session_state.selected_figures:
                    st.write(f"#### {target_plot_name}")
                    st.pyplot(target_fig, use_container_width=True)
                    st.markdown("---")

                # Display other plots in columns
                if other_figs:
                    cols = st.columns(2)
                    col_idx = 0
                    for name, fig in other_figs:
                        if name in st.session_state.selected_figures:
                            with cols[col_idx % len(cols)]:
                                st.write(f"#### {name}")
                                st.pyplot(fig, use_container_width=True)
                            col_idx += 1

    elif not st.session_state.results and data is not None:
        st.info("Configure parameters and click 'Train Model' to see results.")
    elif data is None:
        st.warning("Load data successfully before running the model.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("TensorBoard Direct Access:")
st.sidebar.markdown(f"[Open TensorBoard](http://localhost:{TENSORBOARD_PORT})")