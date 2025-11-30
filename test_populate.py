#!/usr/bin/env python3

import sys
import os

# Add the app-streamlit directory to path
sys.path.insert(0, '/Users/nsls/Documents/Github/nlp-fall-2025/app-streamlit')

# Set workspace path
os.environ['WORKSPACE_PATH'] = '/Users/nsls/Documents/Github/nlp-fall-2025/workspace'

# Initialize workspace
from utils import initialize_workspace
initialize_workspace()

print("Testing populate_job_embeddings...")

try:
    from functions.database import populate_job_embeddings
    success = populate_job_embeddings()
    if success:
        print("SUCCESS: Database populated!")
    else:
        print("FAILED: populate_job_embeddings returned False")
except Exception as e:
    print(f"EXCEPTION: {e}")
    import traceback
    traceback.print_exc()