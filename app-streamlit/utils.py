import streamlit as st
import sys
import os
from dotenv import load_dotenv

def initialize_workspace():
    """
    Initialize workspace path and add NLP workspace functions to Python path.
    This ensures consistent access across all pages.
    """
    # Load environment variables from .env file
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
    
    def get_base_dir():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(current_dir) == 'pages':
            return os.path.dirname(current_dir)
        else:
            return current_dir

    base_dir = get_base_dir()

    # Detect environment and set appropriate path
    if "/usr/src/app" in __file__:
        # Running in Docker container - workspace is mounted at /workspace
        workspace_path = "/workspace"
        # Also add the mounted workspace directory for functions

    else:
        # Running locally
        workspace_path = os.path.join(base_dir, "..", "workspace")

    if os.path.exists(workspace_path):
        sys.path.append(workspace_path)
    else:
        workspace_path = None

    # Set workspace path in session state for global access across pages
    st.session_state.workspace_path = workspace_path