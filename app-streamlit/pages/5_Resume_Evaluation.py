import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import re
import random
import unicodedata
from typing import Optional, List, Dict
from datetime import datetime


from utils import initialize_workspace

# Initialize workspace path and imports
initialize_workspace()

# Page configuration
st.set_page_config(
    page_title="Resume Evaluation - Batch Analysis",
    page_icon="📊",
    layout="wide"
)

# Load global CSS
try:
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles", "app.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

st.title("Resume Evaluation")

# Try to import required libraries
try:
    from functions.nlp_models import (
        generate_local_embedding, build_skill_ner, extract_skill_entities, skill_jaccard_score,
        simple_tokenize,
        load_trained_topic_model, get_document_topics, compute_topic_similarity
    )
    LOCAL_MODELS_AVAILABLE = True
except ImportError:
    LOCAL_MODELS_AVAILABLE = False
    # Fallback tokenization function
    def simple_tokenize(text):
        """Simple tokenization fallback"""
        if pd.isna(text) or not isinstance(text, str):
            return []
        return str(text).split()

# Import skill lists from nlp_config (same source as Resume Matching page)
try:
    from functions.nlp_config import MASTER_SKILL_LIST
except ImportError:
    # Fallback if nlp_config is not available
    try:
        from functions.nlp_models import MASTER_SKILL_LIST
    except ImportError:
        MASTER_SKILL_LIST = []

try:
    from functions.database import create_db_engine, find_similar_jobs_vector
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

# Try to import Ollama for LLM evaluation
try:
    import ollama  # type: ignore
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Try to import transformers for token counting
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Check if services are available
if not LOCAL_MODELS_AVAILABLE:
    st.error("⚠️ Local NLP models not available. Please install required packages.")
if not DATABASE_AVAILABLE:
    st.error("⚠️ Database not available. Please ensure database connection is configured.")
if not PYPDF_AVAILABLE:
    st.error("⚠️ PyPDF not available. Please install pypdf: pip install pypdf")
if not SPACY_AVAILABLE:
    st.warning("⚠️ spaCy not available. Skills-based matching will be disabled.")
if not OLLAMA_AVAILABLE:
    st.warning("⚠️ Ollama not available. LLM-based evaluation will be disabled. Install: pip install ollama")

# Ollama client setup
def _get_ollama_client() -> Optional["ollama.Client"]:  # type: ignore[name-defined]
    """Construct an Ollama client using environment variables"""
    if not OLLAMA_AVAILABLE:
        return None
    
    api_url = (
        os.getenv("OLLAMA_API_URL")
        or os.getenv("OLLAMA_HOST")
        or "http://127.0.0.1:11434"
    )
    
    try:
        return ollama.Client(host=api_url)  # type: ignore[attr-defined]
    except Exception:
        return None

# Get model name from environment
def _get_ollama_model() -> str:
    """Get Ollama model name from environment or use default"""
    return (
        os.getenv("OLLAMA_MODEL")
        or os.getenv("OLLAMA_DEFAULT_MODEL")
        or "gpt-oss:20b"
    )

st.markdown("""
### Overview
This page evaluates all PDF resumes from the `Resume_testing` folder. For each resume, it finds its **top K matching jobs** from the database using the same logic as the Resume Matching page.

**Evaluation Mode**: Each resume is evaluated against its top K matching jobs, resulting in K × the number of resumes in total evaluations.

Each resume-job pair is scored using the same multi-dimensional matching algorithm:
- **Skill Score (Keyword-based)**: Jaccard similarity between resume and job skills
- **Semantic Score**: Cosine similarity of SBERT embeddings
- **Topic Score**: Cosine similarity of LSA 100 topics model distributions (falls back to semantic if model not available)
- **Final Score**: Average(topic, semantic) + (1 - average) × Skill Score
- **LLM Evaluation**: AI-powered Yes/No assessment with:
  - **Recommendations**: If answer is "No", provides specific suggestions to improve the resume
  - **LinkedIn Keywords**: If answer is "No", suggests relevant keywords for LinkedIn job search

All evaluation results are automatically saved to JSON files for further analysis.
""")

# Function to get top 1st job from database (similar to Resume Matching page)
@st.cache_data
def get_top_job() -> Optional[Dict]:
    """Get the top 1st job from the database using the same method as Resume Matching page"""
    if not DATABASE_AVAILABLE:
        return None
    
    try:
        engine = create_db_engine()
        if engine is None:
            return None
        
        with engine.connect() as conn:
            # Get the first job with embedding (same structure as find_similar_jobs_vector)
            # We'll get jobs ordered by ID and take the first one with embedding
            query = text("""
                SELECT id, title, company, text
                FROM jobs
                WHERE embedding IS NOT NULL
                ORDER BY id ASC
                LIMIT 1
            """)
            result = conn.execute(query)
            row = result.fetchone()
            
            if row:
                return {
                    'id': row[0],
                    'title': row[1],
                    'company': row[2],
                    'text': row[3]
                }
            return None
    except Exception as e:
        st.error(f"Error fetching top job: {e}")
        return None

# Alternative: Get top job using find_similar_jobs_vector with a neutral embedding
def get_top_job_via_vector_search() -> Optional[Dict]:
    """Get top job using vector search (similar to Resume Matching) - gets first result"""
    if not DATABASE_AVAILABLE:
        return None
    
    try:
        # Use a neutral/zero embedding to get jobs, then take the first one
        # This mimics how Resume Matching gets jobs
        from functions.nlp_models import generate_local_embedding
        
        # Generate embedding for a neutral query
        neutral_text = "job opportunity"
        neutral_embedding = generate_local_embedding(neutral_text, method="sbert")
        
        if neutral_embedding is not None:
            # Get top 1 job using vector search (same as Resume Matching)
            matching_results = find_similar_jobs_vector(
                neutral_embedding.tolist(),
                embedding_type='sbert',
                top_k=1
            )
            
            if matching_results and len(matching_results) > 0:
                top_job = matching_results[0]
                return {
                    'id': top_job.get('id'),
                    'title': top_job.get('title'),
                    'company': top_job.get('company'),
                    'text': top_job.get('text')
                }
        
        return None
    except Exception as e:
        st.warning(f"Could not get job via vector search: {e}")
        return None

# Function to extract text from PDF
def extract_text_from_pdf(file_path: str) -> Optional[str]:
    """Extract text from PDF file"""
    if not PYPDF_AVAILABLE:
        return None
    
    try:
        pdf_reader = PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.warning(f"Error reading PDF {file_path}: {e}")
        return None

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

# Function to find top matching jobs for a resume (same logic as Resume Matching page)
def find_top_jobs_for_resume(resume_text: str, top_k: int = 3) -> List[Dict]:
    """Find the top matching jobs for a resume using the same logic as Resume Matching page
    
    Args:
        resume_text: Text content of the resume
        top_k: Number of top jobs to return (default 3 for top 3 jobs)
    
    Returns:
        List of top matching job dictionaries with scores, or empty list if no match found
    """
    if not DATABASE_AVAILABLE:
        return []
    
    try:
        # Generate SBERT embedding for resume (same as Resume Matching)
        resume_sbert_emb = generate_local_embedding(resume_text, method="sbert")
        
        if resume_sbert_emb is None:
            return []
        
        # Use database vector search (same as Resume Matching)
        matching_results = find_similar_jobs_vector(
            resume_sbert_emb.tolist(),
            embedding_type='sbert',
            top_k=top_k * 2  # Get more for combined scoring
        )
        
        if not matching_results:
            return []
        
        # Extract resume skills using simple keyword matching (no NER)
        resume_skills = extract_skills_keywords(resume_text, MASTER_SKILL_LIST)
        
        # Load LSA 100 topics model
        topic_model_data = get_lsa_100_topics_model()
        
        # Apply skills scoring to database results (same as Resume Matching)
        enhanced_matches = []
        for job in matching_results:
            job_text = job.get('text', '')
            
            # Extract skills from job text using simple keyword matching (no NER)
            job_skills = extract_skills_keywords(job_text, MASTER_SKILL_LIST)
            
            # Compute scores (same weights as Resume Matching)
            skill_score = skill_jaccard_score(resume_skills, job_skills)
            semantic_score = job['similarity']
            
            # Compute topic score using LSA 100 topics model (fallback to semantic if model not available)
            topic_score = compute_topic_score(resume_text, job_text, topic_model_data)
            if topic_score == 0.0:
                # Fallback to semantic score if topic model not available
                topic_score = semantic_score
            
            # New formula: avg(topic, semantic) + (1 - avg) * Skill_Score
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
        
        # Sort by final score and return top k jobs
        enhanced_matches.sort(key=lambda x: x['final_score'], reverse=True)
        return enhanced_matches[:top_k]
        
    except Exception as e:
        st.warning(f"Error finding top jobs for resume: {e}")
        return []

# Function to evaluate resume against job (kept for compatibility, but now we use find_top_job_for_resume)
def evaluate_resume_against_job(resume_text: str, job: Dict, job_embedding: Optional[np.ndarray]) -> Dict:
    """Evaluate a resume against a job and return scores
    
    Args:
        resume_text: Text content of the resume
        job: Job dictionary with id, title, company, text
        job_embedding: Pre-computed job embedding (None if not available)
    """
    # Extract skills using simple keyword matching (no NER)
    resume_skills = extract_skills_keywords(resume_text, MASTER_SKILL_LIST)
    job_skills = extract_skills_keywords(job['text'], MASTER_SKILL_LIST)
    
    # Calculate skill score
    skill_score = skill_jaccard_score(resume_skills, job_skills)
    
    # Use semantic score from job dict if available (from find_top_job_for_resume)
    if 'semantic_score' in job:
        semantic_score = job['semantic_score']
    else:
        # Fallback: compute semantic score
        semantic_score = 0.0
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            resume_embedding = generate_local_embedding(resume_text, method="sbert")
            if resume_embedding is None:
                raise ValueError("Could not generate resume embedding")
            
            # Use pre-computed job embedding if available, otherwise compute on the fly
            if job_embedding is not None:
                job_emb = job_embedding
            else:
                job_emb_array = generate_local_embedding(job['text'], method="sbert")
                if job_emb_array is None:
                    raise ValueError("Could not generate job embedding")
                job_emb = job_emb_array
            
            # Compute cosine similarity
            resume_emb = np.array(resume_embedding).reshape(1, -1)
            job_emb_reshaped = np.array(job_emb).reshape(1, -1)
            semantic_score = float(cosine_similarity(resume_emb, job_emb_reshaped)[0][0])
        except Exception as e:
            st.warning(f"Error calculating semantic score: {e}")
            semantic_score = 0.0
    
    # Compute topic score using LSA 100 topics model (fallback to semantic if model not available)
    topic_score = compute_topic_score(resume_text, job['text'])
    if topic_score == 0.0:
        # Fallback to semantic score if topic model not available
        topic_score = semantic_score
    
    # New formula: avg(topic, semantic) + (1 - avg) * NER_Score
    avg_topic_semantic = (topic_score + semantic_score) / 2
    final_score = avg_topic_semantic + (1 - avg_topic_semantic) * skill_score
    
    return {
        'skill_score': skill_score,
        'semantic_score': semantic_score,
        'topic_score': topic_score,
        'final_score': final_score,
        'resume_skills': resume_skills,
        'job_skills': job_skills,
        'error': None
    }

# Text cleaning function (same as Data Cleaning page)
def clean_text(text: str) -> str:
    """
    Cleans text (works for both job descriptions and resumes):
    - Removes emails, phone numbers, URLs
    - Removes HTML tags & entities
    - Normalizes bullets and whitespace
    - Collapses extra blank lines
    - Preserves actual content like skills, responsibilities, requirements
    """
    if not isinstance(text, str):
        return ""

    # 1) Unicode normalize + remove zero-width characters
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", text)

    # 2) Remove emails, phone numbers, URLs
    text = re.sub(r"\S+@\S+", " ", text)                          # emails
    text = re.sub(r"\+?\d[\d\-\s\(\)]{7,}\d", " ", text)          # phone numbers
    text = re.sub(r"(https?:\/\/\S+|www\.\S+)", " ", text)        # URLs

    # Remove names like: linkedin jobs, glassdoor jobs, etc.
    text = re.sub(r"(linkedin|glassdoor|indeed|monster|career|company|github)\S*",
                  " ", text, flags=re.IGNORECASE)

    # 3) Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)

    # 4) Normalize bullet points
    text = re.sub(r"[•●▪■◆▶►▸⦿⦾]", "- ", text)
    text = re.sub(r"^-(\S)", r"- \1", text, flags=re.MULTILINE)

    # 5) Normalize dashes
    text = text.replace("–", "-").replace("—", "-")

    # 6) Compact spaces
    text = text.replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)

    # 7) Collapse multiple blank lines (allow max 1)
    lines = [line.strip() for line in text.split("\n")]
    final_lines = []
    blank_seen = False

    for line in lines:
        if line == "":
            if not blank_seen:
                final_lines.append("")
            blank_seen = True
        else:
            final_lines.append(line)
            blank_seen = False

    text = "\n".join(final_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# Token counting and truncation function
@st.cache_resource
def _get_tokenizer():
    """Get tokenizer for counting tokens - cached"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        # Use a common tokenizer that works well for most models
        # GPT-2 tokenizer is fast and widely compatible
        return AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        return None

def count_tokens(text: str) -> int:
    """Count tokens in text using tokenizer if available, otherwise use approximation"""
    if not text:
        return 0
    
    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception:
            pass
    
    # Fallback: approximate token count (roughly 4 characters per token for English)
    return len(text) // 4

def truncate_by_tokens(text: str, max_tokens: int, suffix: str = "...") -> str:
    """Truncate text to fit within max_tokens, preserving word boundaries when possible"""
    if not text:
        return text
    
    tokenizer = _get_tokenizer()
    
    if tokenizer is not None:
        try:
            # Encode the text
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # If text fits, return as is
            if len(tokens) <= max_tokens:
                return text
            
            # Truncate tokens
            truncated_tokens = tokens[:max_tokens]
            
            # Decode back to text
            truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            
            # Add suffix if text was truncated
            if len(tokens) > max_tokens:
                return truncated_text + suffix
            
            return truncated_text
        except Exception:
            pass
    
    # Fallback: character-based truncation (approximate)
    # Roughly 4 characters per token
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    
    truncated = text[:max_chars]
    # Try to cut at word boundary
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.8:  # If we can find a space reasonably close
        truncated = truncated[:last_space]
    
    return truncated + suffix

# Function to evaluate resume match using LLM
def evaluate_with_llm(resume_text: str, job: Dict, model_name: str, 
                      skill_score: float = None, semantic_score: float = None, 
                      topic_score: float = None, final_score: float = None,
                      resume_skills: List[str] = None, job_skills: List[str] = None) -> Dict[str, any]:  # type: ignore
    """Use LLM to evaluate if resume is a good match for the job (Yes/No)
    
    Gets the job description from the database 'text' column and resume text.
    The LLM evaluates the match based solely on the job description and resume content,
    without any pre-computed scores or extracted skills information.
    Returns Yes/No assessment with reasoning, recommendations, and LinkedIn keywords.
    
    Note: Score parameters are kept for backward compatibility but are not used in the prompt.
    """
    if not OLLAMA_AVAILABLE:
        return {
            'llm_match': None,
            'llm_reasoning': None,
            'llm_recommendations': None,
            'linkedin_keywords': None,
            'llm_error': 'Ollama not available'
        }
    
    client = _get_ollama_client()
    if client is None:
        return {
            'llm_match': None,
            'llm_reasoning': None,
            'llm_recommendations': None,
            'linkedin_keywords': None,
            'llm_error': 'Could not connect to Ollama'
        }
    
    # Get job description from database 'text' column
    job_description = job.get('text', '')
    
    # Clean text for better readability (but keep full text, not just tokens)
    cleaned_job_text = clean_text(job_description)
    cleaned_resume_text = clean_text(resume_text)
    
    system_prompt = (
        "You are an expert recruiter evaluating resume-job matches. "
        "Job descriptions often list ideal candidates with all possible skills - this is unrealistic. "
        "Focus on ESSENTIAL qualifications needed to perform core job functions. "
        "Consider transferable skills, related experience, and learning ability. "
        "A 'Yes' means the candidate can reasonably do the job, not that they have every skill listed."
    )
    
    # Token budget allocation (max 5000 tokens total for input)
    # Reserve tokens for: system prompt (~100), user prompt template (~300), job title (~20)
    # Allocate remaining tokens between job description and resume (50/50 split)
    MAX_INPUT_TOKENS = 5000
    system_tokens = count_tokens(system_prompt)
    prompt_template = """Evaluate if this resume matches this job posting. Be practical and realistic.

JOB POSTING:
Title: {title}
Description: {job_text}

RESUME:
{resume_text}

EVALUATION APPROACH:
1. Identify 3-5 CORE skills/requirements essential for this role (ignore nice-to-haves)
2. Check if candidate has these core skills OR transferable/equivalent experience
3. Assess if candidate's experience level aligns with role expectations
4. Consider: Can they reasonably perform the core job functions? If yes → "Yes"

DECISION RULES:
- Answer "Yes" if: Candidate has essential qualifications OR strong transferable skills that indicate they can learn/adapt
- Answer "No" only if: Missing critical core requirements that would prevent basic job performance

OUTPUT FORMAT:
Line 1: "Yes" or "No"
Line 2: One-sentence explanation
If "No", then:
Line 3: "RECOMMENDATIONS:" followed by 3-5 specific, actionable recommendations (one per line, each starting with "-")
Line 4: "LINKEDIN_KEYWORDS:" followed by 5-10 relevant job search keywords (comma-separated)

EXAMPLES:

Yes
The candidate has the essential technical skills and relevant experience level for this role, with transferable capabilities that indicate they can perform effectively.

No
The candidate lacks critical core requirements (e.g., [specific essential skill]) that are fundamental to performing this role.
RECOMMENDATIONS:
- Develop proficiency in [critical missing skill] through projects or training
- Highlight similar technologies or related experience that demonstrates capability
- Obtain certification in [core area] to strengthen qualifications
- Emphasize learning ability and adaptability from past role transitions
LINKEDIN_KEYWORDS:
[relevant keywords, comma-separated]"""
    
    template_tokens = count_tokens(prompt_template.format(title="", job_text="", resume_text=""))
    job_title_tokens = count_tokens(job.get('title', 'N/A'))
    
    # Calculate available tokens for job description and resume
    reserved_tokens = system_tokens + template_tokens + job_title_tokens + 100  # 100 buffer
    available_tokens = MAX_INPUT_TOKENS - reserved_tokens
    
    # Allocate 50/50 between job and resume, but ensure minimum of 500 tokens each
    tokens_per_text = max(500, available_tokens // 2)
    
    # Truncate texts based on token count
    job_text_preview = truncate_by_tokens(cleaned_job_text, tokens_per_text)
    resume_text_preview = truncate_by_tokens(cleaned_resume_text, tokens_per_text)
    
    # Format user prompt using template
    user_prompt = prompt_template.format(
        title=job.get('title', 'N/A'),
        job_text=job_text_preview,
        resume_text=resume_text_preview
    ).strip()
    
    try:
        # Use default context window size (no num_ctx limit)
        # Default context window sizes in Ollama:
        # - Most models: 4,096 tokens (default)
        # - gpt-oss model: 8,192 tokens (default)
        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        
        content = response["message"]["content"].strip()
        
        # Parse response
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Check if it's Yes or No
        is_match = None
        reasoning = None
        recommendations = None
        linkedin_keywords = None
        
        if lines:
            first_line = lines[0].upper()
            if first_line.startswith('YES'):
                is_match = True
            elif first_line.startswith('NO'):
                is_match = False
            
            # Get reasoning (second line)
            if len(lines) > 1:
                reasoning = lines[1]
            
            # Parse recommendations and keywords if answer is No
            if is_match is False:
                recommendations = []
                linkedin_keywords = None
                in_recommendations = False
                in_keywords = False
                
                for i, line in enumerate(lines[2:], start=2):
                    line_upper = line.upper()
                    
                    if line_upper.startswith('RECOMMENDATIONS:'):
                        in_recommendations = True
                        in_keywords = False
                        # Check if recommendations are on the same line
                        if ':' in line and len(line.split(':', 1)) > 1:
                            rec_part = line.split(':', 1)[1].strip()
                            if rec_part.startswith('-'):
                                recommendations.append(rec_part[1:].strip())
                        continue
                    
                    elif line_upper.startswith('LINKEDIN_KEYWORDS:'):
                        in_recommendations = False
                        in_keywords = True
                        # Extract keywords from this line
                        if ':' in line:
                            keywords_str = line.split(':', 1)[1].strip()
                            if keywords_str:
                                linkedin_keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                        # Check next line if keywords are on separate line
                        if not linkedin_keywords and i + 1 < len(lines):
                            next_line = lines[i + 1]
                            if not next_line.upper().startswith('RECOMMENDATIONS:'):
                                linkedin_keywords = [k.strip() for k in next_line.split(',') if k.strip()]
                        break
                    
                    elif in_recommendations:
                        # Collect recommendation lines (lines starting with -)
                        if line.startswith('-'):
                            recommendations.append(line[1:].strip())
                        elif line and not line_upper.startswith('LINKEDIN_KEYWORDS:'):
                            # Sometimes recommendations don't have dashes
                            if len(recommendations) < 5:  # Limit to reasonable number
                                recommendations.append(line)
                    
                    elif in_keywords and not linkedin_keywords:
                        # Keywords might be on next line
                        if ',' in line or ' ' in line:
                            linkedin_keywords = [k.strip() for k in line.split(',') if k.strip()]
                
                # Clean up recommendations (remove empty ones)
                recommendations = [r for r in recommendations if r and len(r.strip()) > 0]
                if not recommendations:
                    recommendations = None
                
                # Clean up keywords
                if linkedin_keywords:
                    linkedin_keywords = [k for k in linkedin_keywords if k and len(k.strip()) > 0]
                    if not linkedin_keywords:
                        linkedin_keywords = None
        
        return {
            'llm_match': is_match,
            'llm_reasoning': reasoning,
            'llm_recommendations': recommendations if recommendations else None,
            'linkedin_keywords': linkedin_keywords,
            'llm_error': None
        }
    except Exception as e:
        return {
            'llm_match': None,
            'llm_reasoning': None,
            'llm_recommendations': None,
            'linkedin_keywords': None,
            'llm_error': f'LLM evaluation error: {str(e)}'
        }


workspace_path = st.session_state.get('workspace_path', None)
if workspace_path:
    resume_dir = os.path.join(workspace_path, "Resume_testing")
else:
    resume_dir = os.path.join("workspace", "Resume_testing")

if not os.path.exists(resume_dir):
    st.error(f"❌ Resume testing directory not found: {resume_dir}")
    st.stop()

# Get all PDF files
pdf_files = sorted([f for f in os.listdir(resume_dir) if f.lower().endswith('.pdf')])

if not pdf_files:
    st.warning(f"⚠️ No PDF files found in {resume_dir}")
    st.stop()


# Load saved evaluation dataset section
st.markdown("---")
st.markdown("### Load Saved Evaluation Dataset")

# Get evaluation output directory
workspace_path = st.session_state.get('workspace_path', None)
if workspace_path:
    evaluation_dir = os.path.join(workspace_path, "Resume_testing", "Evaluation")
else:
    evaluation_dir = os.path.join("workspace", "Resume_testing", "Evaluation")

# Check if evaluation directory exists and get JSON files
json_files = []
if os.path.exists(evaluation_dir):
    json_files = sorted([f for f in os.listdir(evaluation_dir) if f.lower().endswith('.json')], reverse=True)

if json_files:
    
    # File selector
    selected_json = st.selectbox(
        "Select evaluation dataset to load:",
        options=json_files,
        index=0,
        help="Choose a previously saved evaluation dataset to load and view results"
    )
    
    # Show file info
    if selected_json:
        try:
            json_path = os.path.join(evaluation_dir, selected_json)
            file_stats = os.stat(json_path)
            file_size = file_stats.st_size / 1024  # Size in KB
            file_time = datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            # Try to read metadata without loading full file
            with open(json_path, 'r', encoding='utf-8') as f:
                preview_data = json.load(f)
            
            if 'evaluation_timestamp' in preview_data:
                st.caption(f"📅 Created: {preview_data.get('evaluation_timestamp', 'N/A')} | "
                         f"💾 Size: {file_size:.1f} KB | "
                         f"📊 Evaluations: {preview_data.get('total_evaluations', 'N/A')} | "
                         f"📄 Resumes: {preview_data.get('total_resumes', 'N/A')}")
        except Exception:
            pass
    
    if st.button("Load Selected Dataset", use_container_width=True):
        try:
            json_path = os.path.join(evaluation_dir, selected_json)
            with open(json_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            # Extract results from loaded JSON
            if 'results' in loaded_data:
                st.session_state.evaluation_results = loaded_data['results']
                st.session_state.evaluation_json_path = json_path
                
                # Store metadata if available
                if 'evaluation_timestamp' in loaded_data:
                    st.session_state.evaluation_timestamp = loaded_data['evaluation_timestamp']
                if 'total_resumes' in loaded_data:
                    st.session_state.evaluation_total_resumes = loaded_data['total_resumes']
                if 'total_evaluations' in loaded_data:
                    st.session_state.evaluation_total_evaluations = loaded_data['total_evaluations']
                if 'top_jobs_per_resume' in loaded_data:
                    st.session_state.evaluation_top_jobs = loaded_data['top_jobs_per_resume']
                
                st.success(f"✅ Successfully loaded evaluation dataset: {selected_json}")
                st.info(f"📊 Loaded {len(loaded_data['results'])} evaluation results")
                st.rerun()
            else:
                st.error("❌ Invalid evaluation dataset format: missing 'results' field")
        except Exception as e:
            st.error(f"❌ Error loading evaluation dataset: {e}")
else:
    st.info(f"💡 No saved evaluation datasets found in `{evaluation_dir}`. Run an evaluation to create one.")

st.markdown("---")
st.markdown("### Run New Evaluation")

# Sample size selector
max_resumes = len(pdf_files)
sample_size = st.slider(
    "📊 Select number of resumes to evaluate",
    min_value=1,
    max_value=max_resumes,
    value=min(5, max_resumes),  # Default to 5 or total if less than 5
    help=f"Choose how many resumes to process (out of {max_resumes} total). This allows you to test the evaluation on a sample before processing all resumes."
)

# Show which resumes will be processed
if sample_size < max_resumes:
    # Randomly select resumes based on sample size
    selected_files = random.sample(pdf_files, sample_size)
    st.info(f"📋 **{sample_size}** randomly selected resume(s): {', '.join(selected_files)}")
else:
    st.info(f"📋 All **{max_resumes}** resume(s)")

# Top jobs selector
top_jobs_count = st.slider(
    "Select number of top jobs to evaluate per resume",
    min_value=1,
    max_value=10,
    value=3,  # Default to 3
    help="For each resume, the system will find and evaluate against this many top matching jobs from the database."
)

# LLM evaluation toggle
use_llm_evaluation = st.checkbox(
    "Enable LLM-based evaluation (Yes/No match assessment)",
    value=OLLAMA_AVAILABLE,
    disabled=not OLLAMA_AVAILABLE,
    help="Use Ollama LLM to evaluate if each resume is a good match for the job"
)

# Evaluation button
if st.button("Start Evaluation", type="primary", use_container_width=True):
    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Note: Using simple keyword-based skill extraction (no NER)
    # skill_matcher is not needed for keyword-based extraction
    
    # Get LLM model name
    llm_model = _get_ollama_model() if use_llm_evaluation and OLLAMA_AVAILABLE else None
    
    # Store results
    results = []
    
    # Get randomly selected sample of resumes based on sample size
    if sample_size < max_resumes:
        selected_pdf_files = random.sample(pdf_files, sample_size)
    else:
        selected_pdf_files = pdf_files
    
    # Process each resume
    for idx, pdf_file in enumerate(selected_pdf_files):
        pdf_path = os.path.join(resume_dir, pdf_file)
        
        # Update progress
        progress = (idx + 1) / len(selected_pdf_files)
        progress_bar.progress(progress)
        status_text.text(f"Processing {idx + 1}/{len(selected_pdf_files)}: {pdf_file}")
        
        # Extract text from PDF
        resume_text = extract_text_from_pdf(pdf_path)
        
        if resume_text is None or len(resume_text.strip()) == 0:
            result_entry = {
                'resume_file': pdf_file,
                'job_rank': 0,
                'job_title': 'N/A',
                'job_company': 'N/A',
                'job_id': 'N/A',
                'skill_score': 0.0,
                'semantic_score': 0.0,
                'topic_score': 0.0,
                'final_score': 0.0,
                'resume_skills_count': 0,
                'job_skills_count': 0,
                'matching_skills_count': 0,
                'resume_text_length': 0,
                'error': 'Failed to extract text from PDF',
                'llm_match': None,
                'llm_reasoning': None,
                'llm_recommendations': None,
                'linkedin_keywords': None,
                'llm_error': 'Could not evaluate: no resume text'
            }
            results.append(result_entry)
            continue
        
        # Find top N matching jobs for this resume (same logic as Resume Matching page)
        status_text.text(f"Finding top {top_jobs_count} jobs for {idx + 1}/{len(selected_pdf_files)}: {pdf_file}")
        top_jobs = find_top_jobs_for_resume(resume_text, top_k=top_jobs_count)
        
        if not top_jobs:
            result_entry = {
                'resume_file': pdf_file,
                'job_rank': 0,
                'job_title': 'N/A',
                'job_company': 'N/A',
                'job_id': 'N/A',
                'skill_score': 0.0,
                'semantic_score': 0.0,
                'topic_score': 0.0,
                'final_score': 0.0,
                'resume_skills_count': 0,
                'job_skills_count': 0,
                'matching_skills_count': 0,
                'resume_text_length': len(resume_text),
                'error': 'No matching job found in database',
                'llm_match': None,
                'llm_reasoning': None,
                'llm_recommendations': None,
                'linkedin_keywords': None,
                'llm_error': 'Could not find top jobs'
            }
            results.append(result_entry)
            continue
        
        # Evaluate against each of the top N jobs
        for job_rank, top_job in enumerate(top_jobs, start=1):
            # Use scores from find_top_jobs_for_resume (already computed)
            # Calculate matching skills count (using consistent normalization)
            resume_skills = top_job.get('resume_skills', [])
            job_skills = top_job.get('job_skills', [])
            matching_skills = calculate_matching_skills(resume_skills, job_skills)
            
            # LLM evaluation
            llm_result = {
                'llm_match': None, 
                'llm_reasoning': None, 
                'llm_recommendations': None,
                'linkedin_keywords': None,
                'llm_error': None
            }
            if use_llm_evaluation and llm_model:
                status_text.text(f"Evaluating with LLM {idx + 1}/{len(selected_pdf_files)}: {pdf_file} - Job #{job_rank}")
                llm_result = evaluate_with_llm(
                    resume_text, 
                    top_job, 
                    llm_model,
                    skill_score=top_job.get('skill_score'),
                    semantic_score=top_job.get('semantic_score'),
                    topic_score=top_job.get('topic_score'),
                    final_score=top_job.get('final_score'),
                    resume_skills=resume_skills,
                    job_skills=job_skills
                )
            
            result_entry = {
                'resume_file': pdf_file,
                'job_rank': job_rank,
                'job_title': top_job.get('title', 'N/A'),
                'job_company': top_job.get('company', 'N/A'),
                'job_id': top_job.get('id', 'N/A'),
                'skill_score': top_job.get('skill_score', 0.0),
                'semantic_score': top_job.get('semantic_score', 0.0),
                'topic_score': top_job.get('topic_score', 0.0),
                'final_score': top_job.get('final_score', 0.0),
                'resume_skills_count': len(resume_skills),
                'job_skills_count': len(job_skills),
                'matching_skills_count': len(matching_skills),
                'resume_text_length': len(resume_text),
                'resume_skills': resume_skills,
                'job_skills': job_skills,
                'error': None,
                'llm_match': llm_result.get('llm_match'),
                'llm_reasoning': llm_result.get('llm_reasoning'),
                'llm_recommendations': llm_result.get('llm_recommendations'),
                'linkedin_keywords': llm_result.get('linkedin_keywords'),
                'llm_error': llm_result.get('llm_error')
            }
            results.append(result_entry)
    
    # Store results in session state
    st.session_state.evaluation_results = results
    
    # Save results to JSON file
    try:
        workspace_path = st.session_state.get('workspace_path', None)
        if workspace_path:
            output_dir = os.path.join(workspace_path, "Resume_testing", "Evaluation")
        else:
            output_dir = os.path.join("workspace", "Resume_testing", "Evaluation")
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"evaluation_results_{timestamp}.json"
        json_path = os.path.join(output_dir, json_filename)
        
        # Prepare JSON data
        unique_resumes = len(set(r.get('resume_file', '') for r in results))
        json_data = {
            'evaluation_timestamp': timestamp,
            'total_resumes': unique_resumes,
            'total_evaluations': len(results),
            'top_jobs_per_resume': top_jobs_count,
            'evaluation_mode': f'top_{top_jobs_count}_jobs_per_resume',
            'description': f'Each resume is matched with its top {top_jobs_count} matching jobs from database (same logic as Resume Matching page). Total evaluations = {top_jobs_count} × number of resumes.',
            'results': results
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        st.session_state.evaluation_json_path = json_path
        st.info(f"💾 Results saved to: `{json_path}`")
    except Exception as e:
        st.warning(f"Could not save JSON file: {e}")
    
    progress_bar.empty()
    status_text.empty()
    unique_resumes = len(set(r.get('resume_file', '') for r in results))
    st.success(f"✅ Evaluation complete! Processed {unique_resumes} resumes with {len(results)} total evaluations (top {top_jobs_count} jobs per resume).")

# Display results
if st.session_state.get("evaluation_results"):
    st.markdown("---")
    st.markdown("### Evaluation Results")
    
    results = st.session_state.evaluation_results
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Ensure required columns exist with default values
    required_columns = {
        'job_rank': 0,
        'job_title': 'N/A',
        'job_company': 'N/A',
        'job_id': 'N/A',
        'skill_score': 0.0,
        'semantic_score': 0.0,
        'topic_score': 0.0,
        'final_score': 0.0,
        'resume_skills_count': 0,
        'job_skills_count': 0,
        'matching_skills_count': 0,
        'resume_text_length': 0
    }
    
    for col, default_val in required_columns.items():
        if col not in df.columns:
            df[col] = default_val
    
    # Sort by final score descending
    if 'final_score' in df.columns:
        df = df.sort_values('final_score', ascending=False)
    else:
        df = df.sort_index()
    
    # Summary metrics
    unique_resumes = df['resume_file'].nunique() if 'resume_file' in df.columns else len(df)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Resumes", unique_resumes)
        st.caption(f"Total Evaluations: {len(df)}")
    with col2:
        if 'final_score' in df.columns:
            avg_final = df['final_score'].mean()
            st.metric("Avg Final Score", f"{avg_final:.3f}")
        else:
            st.metric("Avg Final Score", "N/A")
    with col3:
        if 'final_score' in df.columns:
            max_final = df['final_score'].max()
            st.metric("Best Match", f"{max_final:.3f}")
        else:
            st.metric("Best Match", "N/A")
    with col4:
        if 'final_score' in df.columns:
            min_final = df['final_score'].min()
            st.metric("Worst Match", f"{min_final:.3f}")
        else:
            st.metric("Worst Match", "N/A")
    with col5:
        if 'llm_match' in df.columns:
            llm_yes = df['llm_match'].sum() if df['llm_match'].notna().any() else 0
            llm_total = df['llm_match'].notna().sum()
            if llm_total > 0:
                st.metric("LLM Yes", f"{llm_yes}/{llm_total}")
            else:
                st.metric("LLM Yes", "N/A")
        else:
            st.metric("LLM Yes", "N/A")
    
    # Score distribution
    if 'final_score' in df.columns:
        st.markdown("#### Score Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(df['final_score'])
        with col2:
            # Score ranges
            high_match = len(df[df['final_score'] >= 0.75])
            medium_match = len(df[(df['final_score'] >= 0.60) & (df['final_score'] < 0.75)])
            low_match = len(df[df['final_score'] < 0.60])
            
            st.markdown("**Match Categories:**")
            st.write(f"🟢 High Match (≥75%): {high_match}")
            st.write(f"🟡 Medium Match (60-75%): {medium_match}")
            st.write(f"🟠 Low Match (<60%): {low_match}")
    
    # Detailed results table
    st.markdown("#### Detailed Results")
    
    # Format scores as percentages for display
    display_df = df.copy()
    
    # Add percentage columns only if the base score columns exist
    if 'final_score' in display_df.columns:
        display_df['Final Score %'] = (display_df['final_score'] * 100).round(2)
    if 'skill_score' in display_df.columns:
        display_df['Skill Score %'] = (display_df['skill_score'] * 100).round(2)
    if 'semantic_score' in display_df.columns:
        display_df['Semantic Score %'] = (display_df['semantic_score'] * 100).round(2)
    if 'topic_score' in display_df.columns:
        display_df['Topic Score %'] = (display_df['topic_score'] * 100).round(2)
    
    # Format LLM match
    if 'llm_match' in display_df.columns:
        display_df['LLM Match'] = display_df['llm_match'].apply(
            lambda x: '✅ Yes' if x is True else '❌ No' if x is False else '⚠️ N/A'
        )
    
    # Select columns for display - only include columns that exist
    base_display_columns = [
        'resume_file', 'job_rank', 'job_title', 'job_company', 
        'Final Score %', 'Skill Score %', 'Semantic Score %', 'Topic Score %',
        'resume_skills_count', 'job_skills_count', 'matching_skills_count'
    ]
    
    # Filter to only include columns that exist in display_df
    display_columns = [col for col in base_display_columns if col in display_df.columns]
    
    # Add LLM match column if available
    if 'LLM Match' in display_df.columns:
        display_columns.append('LLM Match')
    
    # Add overall rank column
    display_df['Rank'] = range(1, len(display_df) + 1)
    display_columns = ['Rank'] + display_columns
    
    # Only display if we have at least some columns
    if display_columns:
        st.dataframe(
            display_df[display_columns],
            use_container_width=True,
            height=600
        )
    else:
        st.warning("⚠️ No displayable columns found in the evaluation results.")
    
    # Expandable details for each resume (grouped by resume, showing all 3 jobs)
    st.markdown("#### Resume Details")
    
    # Group by resume file
    if 'resume_file' in df.columns:
        grouped = df.groupby('resume_file')
    else:
        # If no resume_file column, create a dummy group
        grouped = [(None, df)]
    
    for resume_file, resume_group in grouped:
        # Sort by job_rank if available
        if 'job_rank' in resume_group.columns:
            resume_group = resume_group.sort_values('job_rank')
        
        # Get best score for color coding
        if 'final_score' in resume_group.columns:
            best_score = resume_group['final_score'].max() * 100
        else:
            best_score = 0.0
        
        # Color coding based on best match
        if best_score >= 75:
            color = "🟢"
        elif best_score >= 60:
            color = "🟡"
        else:
            color = "🟠"
        
        num_jobs = len(resume_group)
        resume_name = resume_file if resume_file else "Unknown Resume"
        with st.expander(f"{color} {resume_name} - Top {num_jobs} Matches (Best: {best_score:.1f}%)"):
            # Show all jobs for this resume
            for job_idx, (_, row) in enumerate(resume_group.iterrows(), start=1):
                job_rank = row.get('job_rank', job_idx)
                final_score_pct = row.get('final_score', 0.0) * 100
                
                st.markdown(f"### Job Match #{job_rank} - {final_score_pct:.1f}%")
                
                # Job information header
                if 'job_title' in row and row.get('job_title') != 'N/A':
                    st.markdown(f"**Job:** {row.get('job_title', 'N/A')} at {row.get('job_company', 'N/A')} (ID: {row.get('job_id', 'N/A')})")
                
                # Use single column layout to prevent text cutoff
                st.markdown("**Scores:**")
                if 'final_score' in row:
                    st.write(f"- Final Score: {row.get('final_score', 0.0) * 100:.2f}%")
                if 'skill_score' in row:
                    st.write(f"- Skill Score: {row.get('skill_score', 0.0) * 100:.2f}%")
                if 'semantic_score' in row:
                    st.write(f"- Semantic Score: {row.get('semantic_score', 0.0) * 100:.2f}%")
                if 'topic_score' in row:
                    st.write(f"- Topic Score: {row.get('topic_score', 0.0) * 100:.2f}%")
                
                st.markdown("**Skills:**")
                if 'resume_skills_count' in row:
                    st.write(f"- Resume Skills: {row.get('resume_skills_count', 0)}")
                if 'job_skills_count' in row:
                    st.write(f"- Job Skills: {row.get('job_skills_count', 0)}")
                if 'matching_skills_count' in row:
                    st.write(f"- Matching Skills: {row.get('matching_skills_count', 0)}")
                
                # LLM evaluation
                if 'llm_match' in row and row['llm_match'] is not None:
                    st.markdown("**LLM Evaluation:**")
                    llm_match_text = "✅ Yes" if row['llm_match'] else "❌ No"
                    st.write(f"- Match: {llm_match_text}")
                    if row.get('llm_reasoning'):
                        st.write(f"- Reasoning: {row['llm_reasoning']}")
                    
                    # Show recommendations if answer is No
                    if row['llm_match'] is False and row.get('llm_recommendations'):
                        st.markdown("**📝 Resume Improvement Recommendations:**")
                        for rec in row['llm_recommendations']:
                            st.write(f"  • {rec}")
                    
                    # Show LinkedIn keywords if answer is No
                    if row['llm_match'] is False and row.get('linkedin_keywords'):
                        st.markdown("**🔍 LinkedIn Job Search Keywords:**")
                        keywords_str = ", ".join(row['linkedin_keywords'])
                        st.write(keywords_str)
                        # Make it copyable
                        st.code(keywords_str, language=None)
                elif row.get('llm_error'):
                    st.warning(f"LLM Error: {row['llm_error']}")
                
                if row.get('error'):
                    st.error(f"Error: {row['error']}")
                
                # Add separator between jobs (except for last one)
                if job_idx < len(resume_group):
                    st.markdown("---")
    
    # Export results
    st.markdown("---")
    st.markdown("### Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prepare export DataFrame for CSV
        export_df = df.copy()
        
        # Add score columns only if they exist
        if 'final_score' in export_df.columns:
            export_df['Final Score'] = export_df['final_score']
        if 'skill_score' in export_df.columns:
            export_df['Skill Score'] = export_df['skill_score']
        if 'semantic_score' in export_df.columns:
            export_df['Semantic Score'] = export_df['semantic_score']
        if 'topic_score' in export_df.columns:
            export_df['Topic Score'] = export_df['topic_score']
        
        # Format LLM columns if they exist
        if 'llm_match' in export_df.columns:
            export_df['LLM Match'] = export_df['llm_match'].apply(
                lambda x: 'Yes' if x is True else 'No' if x is False else 'N/A'
            )
        else:
            export_df['LLM Match'] = 'N/A'
        
        # Format recommendations and keywords for CSV (convert lists to strings)
        if 'llm_recommendations' in export_df.columns:
            export_df['LLM Recommendations'] = export_df['llm_recommendations'].apply(
                lambda x: '; '.join(x) if isinstance(x, list) and x else ('N/A' if x is None else str(x))
            )
        else:
            export_df['LLM Recommendations'] = 'N/A'
        
        if 'linkedin_keywords' in export_df.columns:
            export_df['LinkedIn Keywords'] = export_df['linkedin_keywords'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) and x else ('N/A' if x is None else str(x))
            )
        else:
            export_df['LinkedIn Keywords'] = 'N/A'
        
        if 'llm_reasoning' in export_df.columns:
            export_df['LLM Reasoning'] = export_df['llm_reasoning'].apply(
                lambda x: str(x) if x else 'N/A'
            )
        else:
            export_df['LLM Reasoning'] = 'N/A'
        
        export_df['Rank'] = range(1, len(export_df) + 1)
        
        # Base export columns - filter to only include columns that exist
        base_export_columns = [
            'Rank', 'resume_file', 'job_rank', 'job_title', 'job_company', 'job_id',
            'Final Score', 'Skill Score', 
            'Semantic Score', 'Topic Score', 'LLM Match', 'LLM Reasoning',
            'LLM Recommendations', 'LinkedIn Keywords',
            'resume_skills_count', 'job_skills_count', 'matching_skills_count',
            'resume_text_length'
        ]
        
        # Filter to only include columns that exist in export_df
        export_columns = [col for col in base_export_columns if col in export_df.columns]
        
        csv = export_df[export_columns].to_csv(index=False)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"resume_evaluation_results_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Download JSON file
        if st.session_state.get("evaluation_json_path") and os.path.exists(st.session_state.evaluation_json_path):
            with open(st.session_state.evaluation_json_path, 'r', encoding='utf-8') as f:
                json_data = f.read()
            
            st.download_button(
                label="📥 Download Results as JSON",
                data=json_data,
                file_name=os.path.basename(st.session_state.evaluation_json_path),
                mime="application/json",
                use_container_width=True
            )
        else:
            # Generate JSON on the fly if file doesn't exist
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_data = {
                'evaluation_timestamp': timestamp,
                'total_resumes': len(results),
                'evaluation_mode': 'individual_top_job_per_resume',
                'description': 'Each resume is matched with its top matching job from database (same logic as Resume Matching page)',
                'results': results
            }
            
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="📥 Download Results as JSON",
                data=json_str,
                file_name=f"evaluation_results_{timestamp}.json",
                mime="application/json",
                use_container_width=True
            )
    
    # Clear results button
    if st.button("🗑️ Clear Results", use_container_width=True):
        st.session_state.evaluation_results = None
        st.session_state.evaluation_json_path = None
        st.rerun()

# Footer (informational section without custom footer component)
st.markdown("---")
st.markdown("""
### ℹ️ How It Works

1. **Resume Processing**: Extracts text from all PDF files in `workspace/Resume_testing/`
2. **Top Job Matching**: For each resume, finds its top N matching jobs from the database using the same logic as Resume Matching page (N is configurable via slider, range 1-10):
   - Generates SBERT embedding for the resume
   - Uses vector search to find similar jobs
   - Applies skill scoring and combined scoring
   - Returns the top N matching jobs for that resume
3. **Evaluation**: For each resume-job pair, computes:
   - **Skill Score (Keyword-based)**: Jaccard similarity between resume and job skills
   - **Semantic Score**: Cosine similarity of SBERT embeddings
   - **Topic Score**: Cosine similarity of LSA 100 topics model distributions (falls back to semantic if model not available)
   - **Final Score**: Average(topic, semantic) + (1 - average) × Skill Score
4. **LLM Evaluation**: If enabled, uses AI to assess match quality:
   - **Yes/No Assessment**: Determines if resume is a good match
   - **Recommendations**: If "No", provides specific suggestions to improve the resume
   - **LinkedIn Keywords**: If "No", suggests relevant keywords for job searching
5. **Results**: Displays ranked results with detailed metrics, recommendations, and export capability
6. **JSON Export**: All results are automatically saved to JSON files with complete evaluation data

### 📋 Requirements

- **Database**: PostgreSQL with jobs table and embeddings
- **Resume Files**: PDF files in `workspace/Resume_testing/` directory
- **Models**: SBERT for embeddings, spaCy for NER skills extraction

### 💡 Tips

- Results are sorted by Final Score (highest to lowest)
- High match: ≥75%, Medium match: 60-75%, Low match: <60%
- Export results to CSV for further analysis
- Check individual resume details for skill breakdowns
""")
