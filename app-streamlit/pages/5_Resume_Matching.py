import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import tempfile
from typing import Optional, List, Dict

from utils import initialize_workspace

# Initialize workspace path and imports
initialize_workspace()

# Page configuration
st.set_page_config(
    page_title="Resume Matching - Job Search",
    page_icon="📄",
    layout="wide"
)

# Load global CSS
try:
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles", "app.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

st.title("📄 Resume Matching - Find Your Perfect Job")

# Initialize session state
if 'matching_results' not in st.session_state:
    st.session_state.matching_results = None
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = None
if 'resume_embedding' not in st.session_state:
    st.session_state.resume_embedding = None

# Try to import required libraries
try:
    from functions.database import generate_openai_embedding, find_similar_jobs
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# Check if services are available
if not OPENAI_AVAILABLE:
    st.error("⚠️ OpenAI library not available. Please install: `pip install openai`")
if not PYPDF_AVAILABLE:
    st.warning("⚠️ PyPDF not available. PDF upload will not work. Install: `pip install pypdf`")

# Function to extract text from PDF
def extract_text_from_pdf(file) -> Optional[str]:
    """Extract text from PDF file"""
    if not PYPDF_AVAILABLE:
        st.error("PyPDF not available. Please install pypdf: pip install pypdf")
        return None

    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# Function to extract text from TXT
def extract_text_from_txt(file) -> Optional[str]:
    """Extract text from TXT file"""
    try:
        content = file.read().decode("utf-8")
        return content.strip()
    except Exception as e:
        st.error(f"Error reading TXT file: {e}")
        return None

# Function to process resume and find matches
def process_resume_and_match(resume_text: str, top_k: int = 10) -> Optional[List[Dict]]:
    """Process resume text and find matching jobs"""
    if not OPENAI_AVAILABLE:
        st.error("OpenAI library not available. Please install openai: pip install openai")
        return None

    if not resume_text:
        st.error("No resume text provided")
        return None

    # Generate embedding
    with st.spinner("Generating resume embedding..."):
        embedding = generate_openai_embedding(resume_text)

    if not embedding:
        st.error("Failed to generate embedding. Please check your OpenAI API key.")
        return None

    # Store in session state
    st.session_state.resume_embedding = embedding

    # Find similar jobs
    with st.spinner("Finding matching jobs..."):
        similar_jobs = find_similar_jobs(embedding, top_k=top_k)

    if not similar_jobs:
        st.warning("No similar jobs found in database. Please ensure jobs have been inserted.")
        return None

    return similar_jobs

# Main content
st.markdown("""
Upload your resume and find the most relevant job opportunities based on semantic similarity.
The system uses OpenAI embeddings and vector search to match your skills and experience with job requirements.
""")

# File upload section
st.markdown("### 📤 Upload Your Resume")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a resume file",
        type=['pdf', 'txt'],
        help="Upload your resume as PDF or TXT file"
    )

with col2:
    top_k = st.slider(
        "Number of matches to show",
        min_value=5,
        max_value=20,
        value=10,
        help="How many top matching jobs to display"
    )

# Alternative text input
st.markdown("### ✏️ Or Paste Resume Text")
resume_text_input = st.text_area(
    "Paste your resume text here",
    height=200,
    placeholder="Copy and paste your resume content here if you don't have a file...",
    help="Alternatively, paste your resume text directly"
)

# Process resume
if uploaded_file or resume_text_input:
    resume_text = None

    if uploaded_file:
        # Process uploaded file
        file_type = uploaded_file.type

        if file_type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        elif file_type == "text/plain":
            resume_text = extract_text_from_txt(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload PDF or TXT files.")

        if resume_text:
            st.success(f"✅ Resume extracted from {uploaded_file.name} ({len(resume_text)} characters)")

    elif resume_text_input:
        resume_text = resume_text_input.strip()
        if resume_text:
            st.success(f"✅ Resume text loaded ({len(resume_text)} characters)")

    # Store resume text
    if resume_text:
        st.session_state.resume_text = resume_text

        # Show resume preview
        with st.expander("📄 Resume Preview"):
            st.text_area(
                "Resume Content",
                value=resume_text[:2000] + ("..." if len(resume_text) > 2000 else ""),
                height=300,
                disabled=True
            )

        # Find matches button
        if st.button("🔍 Find Matching Jobs", type="primary", use_container_width=True):
            matching_results = process_resume_and_match(resume_text, top_k)

            if matching_results:
                st.session_state.matching_results = matching_results
                st.success(f"✅ Found {len(matching_results)} matching jobs!")
                st.rerun()

# Display results
if st.session_state.matching_results:
    st.markdown("---")
    st.markdown("### 🎯 Top Matching Jobs")

    results = st.session_state.matching_results

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jobs Found", len(results))
    with col2:
        if results:
            avg_similarity = np.mean([job['similarity'] for job in results])
            st.metric("Avg Similarity", f"{avg_similarity:.3f}")
    with col3:
        if results:
            max_similarity = max(job['similarity'] for job in results)
            st.metric("Best Match", f"{max_similarity:.3f}")

    # Display job matches
    for i, job in enumerate(results, 1):
        similarity_percent = job['similarity'] * 100

        # Color coding based on similarity
        if similarity_percent >= 85:
            color = "🟢"  # High match
        elif similarity_percent >= 75:
            color = "🟡"  # Medium match
        else:
            color = "🟠"  # Lower match

        with st.expander(f"{color} Match #{i}: {job.get('title', 'N/A')} at {job.get('company', 'N/A')} - {similarity_percent:.1f}% Match"):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Job Title:** {job.get('title', 'N/A')}")
                st.markdown(f"**Company:** {job.get('company', 'N/A')}")
                st.markdown(f"**Similarity Score:** {similarity_percent:.1f}%")

            with col2:
                # Similarity gauge
                st.metric("Match Strength", f"{similarity_percent:.1f}%")

            st.markdown("**Job Description:**")
            job_text = job.get('text', '')
            if len(job_text) > 1000:
                st.write(job_text[:1000] + "...")
                with st.expander("Read Full Description"):
                    st.write(job_text)
            else:
                st.write(job_text)

    # Export results
    st.markdown("---")
    st.markdown("### 💾 Export Results")

    if st.button("📊 Export to CSV"):
        export_df = pd.DataFrame([{
            'Rank': i+1,
            'Job_ID': job['id'],
            'Title': job.get('title', ''),
            'Company': job.get('company', ''),
            'Similarity': job['similarity'],
            'Similarity_Percent': job['similarity'] * 100
        } for i, job in enumerate(results)])

        csv = export_df.to_csv(index=False)

        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="job_matches.csv",
            mime="text/csv",
            key="download_csv"
        )

# Show embedding info if available
if st.session_state.resume_embedding:
    st.markdown("---")
    st.markdown("### 🔍 Technical Details")
    with st.expander("Embedding Information"):
        embedding = st.session_state.resume_embedding
        st.write(f"**Embedding Dimensions:** {len(embedding)}")
        st.write(f"**Vector Range:** {min(embedding):.4f} to {max(embedding):.4f}")
        st.write(f"**Vector Mean:** {np.mean(embedding):.4f}")
        st.write(f"**Vector Std:** {np.std(embedding):.4f}")

# Footer
st.markdown("---")
st.markdown("""
### ℹ️ How It Works

1. **Upload Resume**: Upload your resume as PDF or TXT, or paste the text directly
2. **AI Processing**: An embedding library generates a high-dimensional embedding of your resume
3. **Vector Search**: PostgreSQL with pgvector finds the most similar job descriptions using cosine similarity
4. **Results**: View ranked job matches with similarity scores and detailed job information

### 📋 Requirements

- **OpenAI API Key**: Set `OPENAI_API_KEY` environment variable
- **Database**: PostgreSQL with pgvector extension and populated jobs table
- **Libraries**: `openai`, `pypdf` (for PDF processing)

### 🔧 Setup Instructions

1. **Database Setup**:
   ```sql
   -- Run the SQL script to create the jobs table
   -- File: create_jobs_table.sql
   ```

2. **Insert Job Data**:
   - Go to NLP Analytics page
   - Load job data and insert into database

3. **Environment Variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   # Database credentials should be in .env file
   ```

### 💡 Tips

- **File Formats**: Both PDF and TXT files are supported
- **Text Quality**: Better formatted resumes produce more accurate matches
- **Similarity Scores**: Higher percentages indicate better matches
- **Export Results**: Download your matches as CSV for further analysis

For technical support or questions, please check the project documentation.
""")

# Add a clear results button
if st.session_state.matching_results:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.matching_results = None
            st.session_state.resume_text = None
            st.session_state.resume_embedding = None
            st.rerun()