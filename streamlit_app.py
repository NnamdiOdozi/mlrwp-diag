"""
streamlit_app.py (Optimized TensorBoard Integration)

Streamlit front-end for the Machine Learning in Reserving - Diagnostic App.
"""


import os
os.environ['STREAMLIT_SERVER_WATCH_MODULES'] = 'false'
os.environ['STREAMLIT_SERVER_WATCHERS_IGNORE_TORCH'] = 'true'

import streamlit as st
import pandas as pd
#import os
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
import sys



# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Get API keys from environment variables
cookie_secret = os.environ.get("cookie_secret")

# Define the log directory and ensure it exists
log_dir = 'app_log'
os.makedirs(log_dir, exist_ok=True)

# Create the full path to the log file
log_filename = os.path.join(log_dir, f"app_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# Setup basic logging
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

# Improved function to check if running on local machine
def is_running_locally():
    hostname = socket.gethostname()
    # Check if hostname contains common local machine identifiers
    return (hostname == "localhost" or 
            hostname.startswith("Nnamdi") or 
            hostname.startswith("LAPTOP-") or
            hostname.lower().startswith("desktop-"))

# Function to get the server's IP address or hostname
def get_server_address():
    """Get the appropriate server address for TensorBoard"""
    # If running locally, use localhost
    if is_running_locally():
        return "localhost"
    
    # For remote servers, try different approaches to get the actual server address
    try:
        # Try to get the server's IP that's exposed to the internet
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Doesn't need to be reachable, just used to get local IP
        s.connect(('8.8.8.8', 1))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        pass
    
    # If that fails, try using the hostname
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        return ip
    except:
        pass
    
    # If all else fails, check if Streamlit gives us the external URL
    try:
        external_url = st.get_option('server.headless') and st.get_option('server.address')
        if external_url and external_url != 'localhost':
            return external_url
    except:
        pass
    
    # Last resort - return localhost, but log the issue
    logging.warning("Could not determine server address, using localhost as fallback")
    return "localhost"

# Function to get the TensorBoard URL
def get_tensorboard_url():
    """Get the appropriate TensorBoard URL based on environment"""
    # Check if running locally
    if is_running_locally():
        return "http://localhost:6006"
    else:
        # When running on the server, use the proxied HTTPS URL
        return "https://mlrwp-diag.uk-ba.net/tensorboard"

# Function to check if TensorBoard is already running
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
    
    # Start TensorBoard process with faster refresh rate
    try:
        process = subprocess.Popen(
            ["tensorboard", "--logdir", TENSORBOARD_LOGDIR, 
             "--port", str(TENSORBOARD_PORT),
             "--bind_all",  # Bind to all interfaces
             "--reload_interval", "0.5",  # Reload every 0.5 seconds
             "--reload_multifile", "true"],  # Enable faster reloading
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

# This code block is to carry out authentication
from streamlit_google_auth import Authenticate

# Initialize the authenticator
authenticator = Authenticate(
    secret_credentials_path='.streamlit/google_credentials.json', 
    cookie_name='mlrwp-diag_auth_session',
    redirect_uri="https://mlrwp-diag.uk-ba.net/oauth2/callback",
    cookie_key=cookie_secret  
)

# Check authentication
authenticator.check_authentification()

# Show login/logout
if not st.session_state['connected']:
    authenticator.login()
else:
    # Your authenticated content here
    st.write(f"Welcome {st.session_state['name']}!")
    st.write(f"Email: {st.session_state['email']}")
    
    # Add logout button
    authenticator.logout()
    
    # Your main app content goes here
    st.write("Your authenticated app content...")

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
if 'training_active' not in st.session_state:
    st.session_state.training_active = False

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

# Auto-refresh during active training
if st.session_state.training_active:
    st.markdown("""
    <meta http-equiv="refresh" content="5">
    """, unsafe_allow_html=True)
    st.info("Auto-refresh is active during training (every 5 seconds)")

# Run model button
if st.button("Train Model and Generate Diagnostics", disabled=(data is None), type="primary"):
    if data is not None:
        with st.status("Starting diagnostic run...", expanded=True) as status:
            try:
                # Set training active flag
                st.session_state.training_active = True
                
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
                status.write(f"- **Final MSE**: {results.get('mse', 'N/A'):,.2f}")

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
                
                # Set training inactive flag
                st.session_state.training_active = False
                
                status.update(label="Diagnostic run completed!", state="complete", expanded=False)
                st.rerun()

            except Exception as e:
                # Set training inactive flag
                st.session_state.training_active = False
                
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
    
    # Get the proper TensorBoard URL based on environment
    tensorboard_url = get_tensorboard_url()
    
    # Check if TensorBoard is running
    if is_port_in_use(TENSORBOARD_PORT):
        # Display the actual URL being used (helpful for debugging)
        st.info(f"TensorBoard URL: {tensorboard_url}")
        
        # Create the iframe for embedded TensorBoard with auto-refresh
        tensorboard_iframe = f"""
        <iframe 
            src="{tensorboard_url}?autorefresh=1&refresh=1"
            width="100%" 
            height="800px" 
            style="border:none;border-radius:5px;"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
            allowfullscreen>
        </iframe>
        """
        
        # Add direct links with different refresh options
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**[Open TensorBoard in new tab]({tensorboard_url})**")
        with col2:
            st.markdown(f"**[Open with auto-refresh]({tensorboard_url}?autorefresh=1)**")
        
        # Add refresh button that users can click manually
        if st.button("Refresh TensorBoard View"):
            st.rerun()
        
        # Embed the TensorBoard UI
        html(tensorboard_iframe, height=800)
        
        # Add instructions
        st.markdown("""
        ### Tips for Real-time Monitoring
        
        If charts aren't updating automatically:
        1. Click the refresh button in TensorBoard (↻ in top right)
        2. Enable auto-refresh using the toggle in TensorBoard settings
        3. Open TensorBoard in a separate window with the auto-refresh link above
        4. Use the 'Refresh TensorBoard View' button to reload the entire frame
        """)
    else:
        st.warning("TensorBoard is not running. Please restart the app or run TensorBoard manually.")
        st.code(f"tensorboard --logdir={TENSORBOARD_LOGDIR} --port={TENSORBOARD_PORT} --bind_all", language="bash")
        st.markdown("Run this command in your terminal to start TensorBoard manually.")

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
tensorboard_url = get_tensorboard_url()
st.sidebar.markdown(f"[Open TensorBoard]({tensorboard_url})")

# This clears the cache for the next user but preserves the current view
if st.session_state.results:
    if st.sidebar.button("Release Memory (Training Complete)"):
    # Clear Streamlit cache
        previous_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        st.cache_data.clear()
        
        # Clear session state variables that might hold large objects
        for key in list(st.session_state.keys()):
            if key.startswith('model_') or key == 'results' or key == 'current_run_figures':
                del st.session_state[key]
        
        # Aggressive garbage collection
        import gc
        
        
        # Run garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # Force memory release to OS (Linux-specific)
        try:
            # Get the current process
            process = psutil.Process(os.getpid())
            
            # For Ubuntu 24.04, use malloc_trim from libc
            import ctypes
            libc = ctypes.CDLL('libc.so.6')
            libc.malloc_trim(0)
            
            # Additional Linux-specific memory pressure technique
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('1')
        except Exception as e:
            st.sidebar.warning(f"Note: Some advanced memory release failed: {str(e)}")
        
        st.sidebar.success(f"Memory released! Current usage: {psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024):.1f} MB, down from {previous_mem:.1f} MB")

