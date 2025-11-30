import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import re
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

# Initialize session state variables used on this page
if "matching_results" not in st.session_state:
    st.session_state.matching_results = None
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None
if "resume_embedding" not in st.session_state:
    st.session_state.resume_embedding = None

# Load global CSS
try:
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles", "app.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

st.title("📄 Resume Matching - Find Your Perfect Job")


# Try to import required libraries
try:
    from functions.nlp_models import (
        load_sbert_model, generate_local_embedding, find_similar_jobs_local,
        compute_job_embeddings_sbert, build_skill_ner, extract_skill_entities, skill_jaccard_score,
        load_spacy_model,
        SENTENCE_TRANSFORMERS_AVAILABLE,
        find_similar_jobs_trained, load_trained_word2vec_model, 
        simple_tokenize, get_doc_embedding_w2v,
        load_trained_topic_model, get_document_topics, compute_topic_similarity
    )
    import torch
    LOCAL_MODELS_AVAILABLE = True
except ImportError:
    LOCAL_MODELS_AVAILABLE = False

# Import skill lists from nlp_config
try:
    from functions.nlp_config import MASTER_SKILL_LIST, EXTRA_SKILLS
except ImportError:
    # Fallback if nlp_config is not available
    MASTER_SKILL_LIST = []
    EXTRA_SKILLS = []

try:
    from functions.database import find_similar_jobs, find_similar_jobs_vector, create_db_engine
    from sqlalchemy import text
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# Try to import spaCy for NER
try:
    import spacy
    from spacy.matcher import PhraseMatcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Note: build_skill_ner, extract_skill_entities, and skill_jaccard_score 
# are imported from functions.nlp_models to ensure consistency across pages

# Simple keyword-based skill extraction (without NER)
def extract_skills_keywords(text: str, skill_list: List[str]) -> List[str]:
    """
    Extract skills from text using simple keyword matching (no NER).
    Checks if skills from the skill list appear in the text (case-insensitive).
    
    Args:
        text: Text to extract skills from
        skill_list: List of skills to search for
    
    Returns:
        List of matching skills (lowercase, deduplicated, sorted)
    """
    if not text or not skill_list:
        return []
    
    text_lower = text.lower()
    skills_found = set()
    
    for skill in skill_list:
        skill_lower = skill.lower().strip()
        if not skill_lower:
            continue
        
        # Check if skill appears in text (word boundary matching for single words,
        # substring matching for multi-word skills)
        if ' ' in skill_lower:
            # Multi-word skill: check if it appears as substring
            if skill_lower in text_lower:
                skills_found.add(skill_lower)
        else:
            # Single-word skill: use word boundary matching
            pattern = r'\b' + re.escape(skill_lower) + r'\b'
            if re.search(pattern, text_lower):
                skills_found.add(skill_lower)
    
    return sorted(list(skills_found))

# Helper function to calculate matching skills consistently
def calculate_matching_skills(resume_skills: List[str], job_skills: List[str]) -> set:
    """
    Calculate matching skills between resume and job, ensuring consistent normalization.
    Normalizes skills to lowercase and removes duplicates before comparison.
    """
    # Normalize skills: convert to lowercase and remove duplicates
    resume_skills_normalized = {skill.lower().strip() for skill in resume_skills if skill}
    job_skills_normalized = {skill.lower().strip() for skill in job_skills if skill}
    # Return intersection
    return resume_skills_normalized & job_skills_normalized

# Helper function to load the specific LSA 100 topics model
@st.cache_resource
def get_lsa_100_topics_model():
    """Load the saved LSA 100 topics model - cached"""
    try:
        import joblib
        workspace_path = st.session_state.get('workspace_path')
        if workspace_path:
            models_dir = os.path.join(workspace_path, "models")
        else:
            models_dir = "models"
        
        model_filename = "topic_model_lsa_100topics.joblib"
        model_path = os.path.join(models_dir, model_filename)
        
        if os.path.exists(model_path):
            try:
                model_data = joblib.load(model_path)
                return {
                    'vectorizer': model_data['vectorizer'],
                    'model': model_data['model'],
                    'results': model_data['results']
                }
            except Exception as e:
                st.warning(f"Error loading LSA 100 topics model: {e}")
                return None
        else:
            st.warning(f"LSA 100 topics model not found at {model_path}")
            return None
    except ImportError:
        st.warning("joblib not available for loading topic model")
        return None
    except Exception:
        return None

# Helper function to compute topic similarity using LDA/LSA (kept for backward compatibility)
@st.cache_resource
def get_topic_model(_method='LDA', _n_topics=10):
    """Load topic model (LDA or LSA) - cached"""
    try:
        return load_trained_topic_model(method=_method, n_topics=_n_topics)
    except Exception:
        return None

def compute_topic_score(resume_text: str, job_text: str, topic_model_data=None) -> float:
    """
    Compute topic similarity score between resume and job using LSA 100 topics model.
    Falls back to 0.0 if topic model is not available.
    
    Args:
        resume_text: Resume text
        job_text: Job description text
        topic_model_data: Pre-loaded topic model data (optional)
    
    Returns:
        Topic similarity score (0.0 to 1.0)
    """
    if topic_model_data is None:
        # Load the specific LSA 100 topics model
        topic_model_data = get_lsa_100_topics_model()
    
    if topic_model_data is None:
        # No topic model available, return 0.0 (will use semantic as fallback)
        return 0.0
    
    try:
        # Get topic distributions for both texts
        resume_topics = get_document_topics(resume_text, topic_model_data)
        job_topics = get_document_topics(job_text, topic_model_data)
        
        if resume_topics is None or job_topics is None:
            return 0.0
        
        # Compute cosine similarity between topic distributions
        topic_score = compute_topic_similarity(resume_topics, job_topics)
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, float(topic_score)))
    except Exception as e:
        # If any error occurs, return 0.0
        return 0.0

# Check if services are available
if not LOCAL_MODELS_AVAILABLE:
    st.error("⚠️ Local NLP models not available. Please install required packages: pip install sentence-transformers spacy && python -m spacy download en_core_web_sm")
if not SPACY_AVAILABLE:
    st.warning("⚠️ spaCy not available. Skills-based matching will be disabled. Install: pip install spacy && python -m spacy download en_core_web_sm")

st.markdown("### 🤖 Manual Model Loading")

# Check MPS availability
if LOCAL_MODELS_AVAILABLE:
    try:
        # More robust MPS detection
        mps_available = False
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
            if torch.backends.mps.is_built():
                try:
                    # Try to create MPS device
                    device = torch.device('mps')
                    # Test with a small tensor
                    test_tensor = torch.randn(1).to(device)
                    mps_available = True
                except:
                    mps_available = False
        
        
        if not mps_available:
            st.info("💡 **MPS Troubleshooting:** If you have an M2/M3 Mac, try restarting your terminal and Streamlit. MPS requires macOS 12.3+ and PyTorch 1.12+.")
            
    except Exception as e:
        st.info(f"Debug: Error checking MPS: {e}")


st.markdown("**Model Status:**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    sbert_loaded = st.session_state.get('sbert_loaded', False)
    st.write(f"SBERT: {'Loaded' if sbert_loaded else 'Not Loaded'}")
    if st.button("Load SBERT Model"):
        with st.spinner("Loading SBERT model..."):
            try:
                from functions.nlp_models import load_sbert_model
                model = load_sbert_model()
                if model:
                    st.session_state.sbert_loaded = True
                    st.success("SBERT model loaded successfully!")
                    st.rerun()
                else:
                    st.error("Failed to load SBERT model")
            except Exception as e:
                st.error(f"Error: {e}")

with col2:
    w2v_loaded = st.session_state.get('w2v_loaded', False)
    st.write(f"Word2Vec: {'Loaded' if w2v_loaded else 'Not Loaded'}")
    if st.button("Load Word2Vec Model"):
        with st.spinner("Loading Word2Vec model..."):
            try:
                from functions.nlp_models import load_trained_word2vec_model
                model = load_trained_word2vec_model()
                if model:
                    st.session_state.w2v_loaded = True
                    st.success("Word2Vec model loaded successfully!")
                    st.rerun()
                else:
                    st.error("Failed to load Word2Vec model")
            except Exception as e:
                st.error(f"Error: {e}")

with col3:
    spacy_loaded = st.session_state.get('spacy_loaded', False)
    st.write(f"spaCy NER: {'Loaded' if spacy_loaded else 'Not Loaded'}")
    if st.button("Load spaCy NER"):
        with st.spinner("Loading spaCy model..."):
            try:
                nlp = load_spacy_model()
                if nlp:
                    st.session_state.spacy_loaded = True
                    st.success("spaCy NER model loaded successfully!")
                    st.rerun()
                else:
                    st.error("Failed to load spaCy model")
            except Exception as e:
                st.error(f"Error: {e}")

with col4:
    skill_ner_loaded = st.session_state.get('skill_ner_loaded', False)
    st.write(f"Skill NER: {'Built' if skill_ner_loaded else 'Not Built'}")
    if st.button("Build Skill NER"):
        with st.spinner("Building Skill NER matcher..."):
            try:
                matcher = build_skill_ner(MASTER_SKILL_LIST)
                if matcher:
                    st.session_state.skill_ner_loaded = True
                    st.success("Skill NER built successfully!")
                    st.rerun()
                else:
                    st.error("Failed to build Skill NER")
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")

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
@st.cache_data
def load_and_prepare_job_data():
    """Load and prepare job data for matching (cached)"""
    try:
        import json
        workspace_path = st.session_state.get('workspace_path', '/workspace')
        json_path = os.path.join(workspace_path, "Data", "combined_data.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                job_data = json.load(f)
            jobs_df = pd.DataFrame(job_data)
        else:
            st.error("Job data not found. Please ensure combined_data.json exists in workspace/Data/")
            return None, None
    except Exception as e:
        st.error(f"Error loading job data: {e}")
        return None, None

    # Filter out jobs without text
    valid_jobs = jobs_df[jobs_df['Description'].notna() & (jobs_df['Description'] != '')].copy()
    if len(valid_jobs) == 0:
        st.error("No valid job data found")
        return None, None

    job_texts = valid_jobs['Description'].tolist()
    return job_texts, valid_jobs

def process_resume_and_match(resume_text: str, top_k: int = 10) -> Optional[List[Dict]]:

    # Load job data (cached)
    job_texts, valid_jobs = load_and_prepare_job_data()

    if job_texts is None or valid_jobs is None:
        return None

    # Always use pre-trained SBERT with database vector search
    with st.spinner("Finding matches using SBERT vector search..."):
        # Generate SBERT embedding for resume
        resume_sbert_emb = generate_local_embedding(resume_text, method="sbert")
        
        if resume_sbert_emb is not None and DATABASE_AVAILABLE:
            # Use database vector search (fast!)
            try:
                matching_results = find_similar_jobs_vector(
                    resume_sbert_emb.tolist(), 
                    embedding_type='sbert', 
                    top_k=top_k * 2  # Get more for combined scoring
                )
                
                if matching_results:
                    # Apply skills scoring to database results using simple keyword matching (no NER)
                    resume_skills = extract_skills_keywords(resume_text, MASTER_SKILL_LIST)
                    
                    # Load LSA 100 topics model once for all jobs
                    topic_model_data = get_lsa_100_topics_model()
                    
                    enhanced_matches = []
                    for job in matching_results:
                        job_text = job.get('text', '')
                        
                        # Extract skills from job text using simple keyword matching (no NER)
                        job_skills = extract_skills_keywords(job_text, MASTER_SKILL_LIST)
                        
                        # Compute scores
                        skill_score = skill_jaccard_score(resume_skills, job_skills)
                        semantic_score = job['similarity']
                        
                        # Compute topic score using LSA 100 topics model (fallback to semantic if model not available)
                        topic_score = compute_topic_score(resume_text, job_text, topic_model_data)
                        if topic_score == 0.0:
                            # Fallback to semantic score if topic model not available
                            topic_score = semantic_score
                        
                        # New formula: avg(topic, semantic) + (1 - avg) * NER_Score
                        avg_topic_semantic = (topic_score + semantic_score) / 2
                        final_score = avg_topic_semantic + (1 - avg_topic_semantic) * skill_score
                        
                        enhanced_job = job.copy()
                        enhanced_job.update({
                            'skill_score': skill_score,
                            'semantic_score': semantic_score,
                            'topic_score': topic_score,
                            'final_score': final_score,
                            'resume_skills': resume_skills,
                            'job_skills': job_skills
                        })
                        enhanced_matches.append(enhanced_job)
                    
                    # Sort by final score and return top_k
                    enhanced_matches.sort(key=lambda x: x['final_score'], reverse=True)
                    return enhanced_matches[:top_k]
                else:
                    st.warning("No matches found in database. Please ensure jobs have been indexed.")
                    return []
                    
            except Exception as e:
                st.warning(f"Database search failed: {e}. Falling back to on-demand computation...")
        
        # Fallback: Original on-demand computation method
        st.info("Using on-demand embedding computation (slower)...")
        # ... [rest of original code for fallback]

# Main content
st.markdown("""
Upload your resume and find the most relevant job opportunities based on **combined scoring** that balances:
- **Skills Match (Keyword-based)**: Technical and soft skills alignment using keyword matching
- **Semantic Similarity**: Contextual meaning using SBERT embeddings  
- **Topic Relevance**: Thematic alignment using LSA 100 topics model
- **Final Score**: Average(topic, semantic) + (1 - average) × Skill Score

The system uses Sentence-BERT (SBERT) embeddings for efficient similarity search, LSA 100 topics model for topic modeling, and keyword-based skill extraction.
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

        # Show resume analysis
        with st.expander("🔍 Resume Analysis"):
            # Extract and display skills using simple keyword matching (no NER)
            resume_skills = extract_skills_keywords(resume_text, MASTER_SKILL_LIST)
            
            st.markdown("**📋 Extracted Skills (Keyword-based):**")
            if resume_skills:
                st.write(f"Found {len(resume_skills)} skills:")
                st.write(", ".join(resume_skills))
            else:
                st.write("No skills extracted.")
            
            # Word embeddings
            st.markdown("**🧠 Word Embeddings:**")
            
            # SBERT
            resume_sbert_emb = generate_local_embedding(resume_text, method="sbert")
            if resume_sbert_emb is not None:
                # Store embedding for later technical details section
                st.session_state.resume_embedding = resume_sbert_emb
                st.write("**SBERT Embedding:**")
                st.write(f"- Dimensions: {len(resume_sbert_emb)}")
                st.write(f"- Range: {min(resume_sbert_emb):.4f} to {max(resume_sbert_emb):.4f}")
                st.write(f"- Mean: {np.mean(resume_sbert_emb):.4f}")
                st.write(f"- Std: {np.std(resume_sbert_emb):.4f}")
            else:
                st.write("SBERT embedding not available.")
            
            # Word2Vec
            w2v_model = load_trained_word2vec_model()
            if w2v_model:
                resume_tokens = simple_tokenize(resume_text)
                resume_w2v_emb = get_doc_embedding_w2v(resume_tokens, w2v_model)
                if resume_w2v_emb is not None:
                    st.write("**Word2Vec Embedding:**")
                    st.write(f"- Dimensions: {len(resume_w2v_emb)}")
                    st.write(f"- Range: {min(resume_w2v_emb):.4f} to {max(resume_w2v_emb):.4f}")
                    st.write(f"- Mean: {np.mean(resume_w2v_emb):.4f}")
                    st.write(f"- Std: {np.std(resume_w2v_emb):.4f}")
                else:
                    st.write("Word2Vec embedding not available.")
            else:
                st.write("Word2Vec model not loaded.")
            
            # Topics
            st.markdown("**📊 Topic Analysis:**")
            topic_model = get_lsa_100_topics_model()
            if topic_model:
                st.write(f"Topic Model: LSA with 100 topics")
            else:
                st.write("LSA 100 topics model not available. Topic score will use semantic similarity as fallback.")

        # Find matches button
        if st.button("🔍 Find Matching Jobs", type="primary", use_container_width=True):
            matching_results = process_resume_and_match(resume_text, top_k)

            if matching_results:
                st.session_state.matching_results = matching_results
                st.success(f"✅ Found {len(matching_results)} matching jobs!")
                st.rerun()

# Display results
if st.session_state.get("matching_results"):
    st.markdown("---")
    st.markdown("### 🎯 Top Matching Jobs")

    results = st.session_state.matching_results

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jobs Found", len(results))
    with col2:
        if results:
            avg_final = np.mean([job['final_score'] for job in results])
            st.metric("Avg Final Score", f"{avg_final:.3f}")
    with col3:
        if results:
            max_final = max(job['final_score'] for job in results)
            st.metric("Best Match", f"{max_final:.3f}")

    # Display job matches
    for i, job in enumerate(results, 1):
        final_score_percent = job['final_score'] * 100

        # Color coding based on final score
        if final_score_percent >= 75:
            color = "🟢"  # High match
        elif final_score_percent >= 60:
            color = "🟡"  # Medium match
        else:
            color = "🟠"  # Lower match

        with st.expander(f"{color} Match #{i}: {job.get('title', 'N/A')} at {job.get('company', 'N/A')} - {final_score_percent:.1f}% Match"):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Job Title:** {job.get('title', 'N/A')}")
                st.markdown(f"**Company:** {job.get('company', 'N/A')}")
                st.markdown(f"**Final Score:** {final_score_percent:.1f}%")
                
                # Show component scores
                skill_pct = job['skill_score'] * 100
                semantic_pct = job['semantic_score'] * 100
                topic_pct = job['topic_score'] * 100
                st.markdown(f"**Skill Score:** {skill_pct:.1f}% | **Semantic Score:** {semantic_pct:.1f}% | **Topic Score:** {topic_pct:.1f}%")

            with col2:
                # Final score gauge
                st.metric("Match Strength", f"{final_score_percent:.1f}%")

            st.markdown("**Job Description:**")
            job_text = job.get('text', '')
            if len(job_text) > 1000:
                st.write(job_text[:1000] + "...")
                with st.expander("Read Full Description"):
                    st.write(job_text)
            else:
                st.write(job_text)

            # Show skills comparison
            if job.get('resume_skills') and job.get('job_skills'):
                with st.expander("Skills Analysis"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Resume Skills:**")
                        st.write(", ".join(job['resume_skills'][:10]))  # Show first 10
                        if len(job['resume_skills']) > 10:
                            st.write(f"... and {len(job['resume_skills']) - 10} more")
                    
                    with col2:
                        st.markdown("**Job Skills:**")
                        st.write(", ".join(job['job_skills'][:10]))  # Show first 10
                        if len(job['job_skills']) > 10:
                            st.write(f"... and {len(job['job_skills']) - 10} more")
                    
                    # Show overlapping skills (using consistent normalization)
                    overlap = calculate_matching_skills(
                        job.get('resume_skills', []), 
                        job.get('job_skills', [])
                    )
                    if overlap:
                        st.markdown("**Matching Skills:**")
                        st.write(", ".join(sorted(overlap)))

    # Export results
    st.markdown("---")
    st.markdown("### 💾 Export Results")

    if st.button("📊 Export to CSV"):
        export_df = pd.DataFrame([{
            'Rank': i+1,
            'Job_ID': job['id'],
            'Title': job.get('title', ''),
            'Company': job.get('company', ''),
            'Final_Score': job['final_score'],
            'Final_Score_Percent': job['final_score'] * 100,
            'Skill_Score': job['skill_score'],
            'Semantic_Score': job['semantic_score'],
            'Topic_Score': job['topic_score'],
            'Resume_Skills_Count': len(job.get('resume_skills', [])),
            'Job_Skills_Count': len(job.get('job_skills', [])),
            'Matching_Skills_Count': len(calculate_matching_skills(
                job.get('resume_skills', []), 
                job.get('job_skills', [])
            ))
        } for i, job in enumerate(results)])

        csv = export_df.to_csv(index=False)

        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="job_matches_combined.csv",
            mime="text/csv",
            key="download_csv"
        )

# Show embedding info if available
if st.session_state.get("resume_embedding") is not None:
    st.markdown("---")
    st.markdown("### 🔍 Technical Details")
    with st.expander("Embedding & Skills Information"):
        embedding = st.session_state.resume_embedding
        st.write(f"**Embedding Model:** Sentence-BERT (all-MiniLM-L6-v2)")
        st.write(f"**Embedding Dimensions:** {len(embedding)}")
        st.write(f"**Vector Range:** {min(embedding):.4f} to {max(embedding):.4f}")
        st.write(f"**Vector Mean:** {np.mean(embedding):.4f}")
        st.write(f"**Vector Std:** {np.std(embedding):.4f}")
        
        if st.session_state.get("matching_results"):
            results = st.session_state.matching_results
            if results and 'resume_skills' in results[0]:
                resume_skills = results[0]['resume_skills']
                st.write(f"**Resume Skills Extracted:** {len(resume_skills)}")
                if resume_skills:
                    st.write("**Sample Skills:**", ", ".join(resume_skills[:5]))
                    if len(resume_skills) > 5:
                        st.write(f"... and {len(resume_skills) - 5} more")

# Footer
st.markdown("---")
st.markdown("""
### ℹ️ How It Works

1. **Upload Resume**: Upload your resume as PDF or TXT, or paste the text directly
2. **AI Processing**: Generate SBERT embeddings and extract skills using spaCy NER
3. **Multi-dimensional Matching**: Compute three similarity scores:
   - **Skill Score (Keyword-based)**: Jaccard similarity between resume and job skills
   - **Semantic Score**: Cosine similarity of SBERT embeddings
   - **Topic Score**: Cosine similarity of LSA 100 topics model distributions (falls back to semantic if model not available)
4. **Combined Scoring**: Final Score = Average(topic, semantic) + (1 - average) × Skill Score
5. **Results**: View ranked job matches with detailed component scores and skills analysis

### 📋 Requirements

- **Local Models**: `sentence-transformers` for SBERT embeddings
- **spaCy**: For NER and skills extraction (`en_core_web_sm` model)
- **Job Data**: `combined_data.json` in `workspace/Data/` directory
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `streamlit`

### 🔧 Setup Instructions

1. **Install Required Packages**:
   ```bash
   pip install sentence-transformers spacy scikit-learn
   python -m spacy download en_core_web_sm
   ```

2. **Prepare Job Data**:
   - Ensure `combined_data.json` exists in `workspace/Data/`
   - Job data should have `text`, `title`, `company`, and `id` fields

3. **Run the Application**:
   - The system will automatically load SBERT model on first use
   - Embeddings are computed in real-time (consider pre-computing for production)

### 💡 Tips

- **File Formats**: Both PDF and TXT files are supported
- **Text Quality**: Better formatted resumes produce more accurate matches
- **Combined Scoring**: The final score balances skills expertise, semantic relevance, and topical alignment
- **Skills Analysis**: Expand the skills section to see detailed resume-job skill matching
- **Performance**: SBERT embeddings provide high-quality semantic matching without API costs

**Note**: Topic Score uses LSA 100 topics model (`topic_model_lsa_100topics.joblib`) if available in the `models/` directory. If no topic model is found, it falls back to semantic similarity.

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