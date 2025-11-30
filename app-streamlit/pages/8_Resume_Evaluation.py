import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import re
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

st.title("📊 Resume Evaluation - Batch Analysis")

# Try to import required libraries
try:
    from functions.nlp_models import (
        generate_local_embedding, build_skill_ner, extract_skill_entities, skill_jaccard_score,
        MASTER_SKILL_LIST, simple_tokenize
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
This page evaluates all PDF resumes from the `Resume_testing` folder. For each resume, it finds its **top matching job** from the database using the same logic as the Resume Matching page.

Each resume is scored using the same multi-dimensional matching algorithm:
- **Skill Score (NER)**: Jaccard similarity between resume and job skills
- **Semantic Score**: Cosine similarity of SBERT embeddings
- **Topic Score**: Currently uses semantic score as proxy
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

# Function to find top matching job for a resume (same logic as Resume Matching page)
def find_top_job_for_resume(resume_text: str, skill_matcher, top_k: int = 1) -> Optional[Dict]:
    """Find the top matching job for a resume using the same logic as Resume Matching page
    
    Args:
        resume_text: Text content of the resume
        skill_matcher: spaCy skill matcher
        top_k: Number of top jobs to return (default 1 for top job)
    
    Returns:
        Top matching job dictionary with scores, or None if no match found
    """
    if not DATABASE_AVAILABLE:
        return None
    
    try:
        # Generate SBERT embedding for resume (same as Resume Matching)
        resume_sbert_emb = generate_local_embedding(resume_text, method="sbert")
        
        if resume_sbert_emb is None:
            return None
        
        # Use database vector search (same as Resume Matching)
        matching_results = find_similar_jobs_vector(
            resume_sbert_emb.tolist(),
            embedding_type='sbert',
            top_k=top_k * 2  # Get more for combined scoring
        )
        
        if not matching_results:
            return None
        
        # Extract resume skills
        resume_skills = extract_skill_entities(resume_text, skill_matcher) if skill_matcher else []
        
        # Apply skills scoring to database results (same as Resume Matching)
        enhanced_matches = []
        for job in matching_results:
            job_text = job.get('text', '')
            
            # Extract skills from job text
            job_skills = extract_skill_entities(job_text, skill_matcher) if skill_matcher else []
            
            # Compute scores (same weights as Resume Matching)
            skill_score = skill_jaccard_score(resume_skills, job_skills)
            semantic_score = job['similarity']
            topic_score = semantic_score  # Placeholder (same as Resume Matching)
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
        
        # Sort by final score and return top job (same as Resume Matching)
        enhanced_matches.sort(key=lambda x: x['final_score'], reverse=True)
        top_job = enhanced_matches[0] if enhanced_matches else None
        
        return top_job
        
    except Exception as e:
        st.warning(f"Error finding top job for resume: {e}")
        return None

# Function to evaluate resume against job (kept for compatibility, but now we use find_top_job_for_resume)
def evaluate_resume_against_job(resume_text: str, job: Dict, job_embedding: Optional[np.ndarray], skill_matcher) -> Dict:
    """Evaluate a resume against a job and return scores
    
    Args:
        resume_text: Text content of the resume
        job: Job dictionary with id, title, company, text
        job_embedding: Pre-computed job embedding (None if not available)
        skill_matcher: spaCy skill matcher
    """
    # Extract skills
    resume_skills = extract_skill_entities(resume_text, skill_matcher) if skill_matcher else []
    job_skills = extract_skill_entities(job['text'], skill_matcher) if skill_matcher else []
    
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
    
    # Topic score (using semantic as proxy for now, same as Resume_Matching.py)
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

# Function to evaluate resume match using LLM
def evaluate_with_llm(resume_text: str, job: Dict, model_name: str, 
                      skill_score: float = None, semantic_score: float = None, 
                      topic_score: float = None, final_score: float = None,
                      resume_skills: List[str] = None, job_skills: List[str] = None) -> Dict[str, any]:  # type: ignore
    """Use LLM to evaluate if resume is a good match for the job (Yes/No)
    
    Gets the job description from the database 'text' column, and provides computed scores
    and extracted skills to help the LLM make an informed decision aligned with the scoring system.
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
    
    # Prepare scores information
    scores_info = ""
    if final_score is not None:
        skill_str = f"{skill_score:.3f} ({skill_score*100:.1f}%)" if skill_score is not None else "N/A"
        semantic_str = f"{semantic_score:.3f} ({semantic_score*100:.1f}%)" if semantic_score is not None else "N/A"
        topic_str = f"{topic_score:.3f} ({topic_score*100:.1f}%)" if topic_score is not None else "N/A"
        
        scores_info = f"""
COMPUTED MATCHING SCORES:
- Final Score: {final_score:.3f} ({final_score*100:.1f}%)
- Skill Score (NER): {skill_str}
- Semantic Score: {semantic_str}
- Topic Score: {topic_str}

SCORING GUIDELINES:
- Final Score ≥ 0.75 (75%): Strong match - typically answer "Yes"
- Final Score 0.60-0.75 (60-75%): Moderate match - consider context, usually "Yes" if skills align well
- Final Score < 0.60 (60%): Weak match - typically answer "No"
- The Final Score is calculated as: Average(topic, semantic) + (1 - average) × Skill Score
- Higher scores indicate better alignment between resume and job requirements
"""
    
    # Prepare skills information
    skills_info = ""
    if resume_skills and job_skills:
        matching_skills = set(resume_skills) & set(job_skills)
        skills_info = f"""
EXTRACTED SKILLS ANALYSIS:
- Resume Skills Found: {len(resume_skills)} skills
- Job Required Skills: {len(job_skills)} skills  
- Matching Skills: {len(matching_skills)} skills

Resume Skills: {', '.join(resume_skills[:20])}{'...' if len(resume_skills) > 20 else ''}
Job Required Skills: {', '.join(job_skills[:20])}{'...' if len(job_skills) > 20 else ''}
Matching Skills: {', '.join(sorted(matching_skills)[:20])}{'...' if len(matching_skills) > 20 else ''}
"""
    
    system_prompt = (
        "You are an expert recruiter and hiring manager with access to AI-powered matching scores. "
        "Your task is to evaluate whether a candidate's resume is a good match for a job posting. "
        "You will receive the full job description, resume text, computed matching scores, and extracted skills. "
        "Use the provided scores as a strong indicator, but also consider the overall context, experience level, "
        "and alignment of qualifications. Your evaluation should generally align with the Final Score, but you may "
        "adjust based on critical missing requirements or exceptional qualifications not captured in the scores. "
        "If the answer is 'No', provide specific, actionable recommendations to improve the resume."
    )
    
    # Truncate long texts for LLM (keep first 3000 chars of each)
    job_text_preview = cleaned_job_text[:3000] + ("..." if len(cleaned_job_text) > 3000 else "")
    resume_text_preview = cleaned_resume_text[:3000] + ("..." if len(cleaned_resume_text) > 3000 else "")
    
    user_prompt = f"""
Evaluate if this resume is a good match for this job posting. Use the provided matching scores as your primary guide, but also consider the full context.

JOB POSTING:
Title: {job.get('title', 'N/A')}
Company: {job.get('company', 'N/A')}
Job ID: {job.get('id', 'N/A')}

Job Description:
{job_text_preview}

RESUME:
{resume_text_preview}

{scores_info}

{skills_info}

EVALUATION CRITERIA:
1. Primary indicator: Final Score (higher = better match)
2. Consider skill alignment (matching skills vs required skills)
3. Consider semantic/topic similarity (how well the content aligns)
4. Consider critical missing requirements that might not be captured in scores
5. Consider exceptional qualifications that might boost the match beyond the score

Respond with the following format:
1. First line: "Yes" or "No" (only one word)
2. Second line: A brief one-sentence explanation of your decision, referencing the Final Score
3. If answer is "No", add a third line starting with "RECOMMENDATIONS:" followed by 3-5 specific recommendations to improve the resume (one per line, each starting with "-")
4. If answer is "No", add a fourth line starting with "LINKEDIN_KEYWORDS:" followed by 5-10 relevant keywords for LinkedIn job search (comma-separated)

Example format for "Yes" (high score):
Yes
The Final Score of 85.2% indicates a strong match, with good alignment in skills, experience, and qualifications.

Example format for "No" (low score):
No
The Final Score of 45.3% indicates a weak match, with significant gaps in required skills and experience.
RECOMMENDATIONS:
- Add experience with [specific missing skill/technology]
- Highlight relevant projects or achievements
- Include certifications or training in [area]
- Emphasize transferable skills from related experience
- Add keywords from the job description that are missing
LINKEDIN_KEYWORDS:
[relevant job search keywords, comma-separated]
""".strip()
    
    try:
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

# Evaluation mode explanation
st.markdown("---")
st.markdown("### 🎯 Evaluation Mode")

st.info("""
**How it works:**
- For each resume, the system finds its **top matching job** using the same logic as the Resume Matching page
- Each resume is evaluated against its own top matching job (not a single job for all resumes)
- This ensures each resume is matched with the most relevant job for that specific candidate
""")

# Get resume testing directory
st.markdown("---")
st.markdown("### 📁 Resume Files")

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

st.info(f"Found **{len(pdf_files)}** PDF resume files in `{resume_dir}`")

# LLM evaluation toggle
use_llm_evaluation = st.checkbox(
    "🤖 Enable LLM-based evaluation (Yes/No match assessment)",
    value=OLLAMA_AVAILABLE,
    disabled=not OLLAMA_AVAILABLE,
    help="Use Ollama LLM to evaluate if each resume is a good match for the job"
)

# Evaluation button
if st.button("🚀 Start Evaluation", type="primary", use_container_width=True):
    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Build skill matcher
    skill_matcher = build_skill_ner(MASTER_SKILL_LIST) if SPACY_AVAILABLE else None
    
    # Get LLM model name
    llm_model = _get_ollama_model() if use_llm_evaluation and OLLAMA_AVAILABLE else None
    
    # Store results
    results = []
    
    # Process each resume
    for idx, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(resume_dir, pdf_file)
        
        # Update progress
        progress = (idx + 1) / len(pdf_files)
        progress_bar.progress(progress)
        status_text.text(f"Processing {idx + 1}/{len(pdf_files)}: {pdf_file}")
        
        # Extract text from PDF
        resume_text = extract_text_from_pdf(pdf_path)
        
        if resume_text is None or len(resume_text.strip()) == 0:
            result_entry = {
                'resume_file': pdf_file,
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
        
        # Find top matching job for this resume (same logic as Resume Matching page)
        status_text.text(f"Finding top job for {idx + 1}/{len(pdf_files)}: {pdf_file}")
        top_job = find_top_job_for_resume(resume_text, skill_matcher, top_k=1)
        
        if top_job is None:
            result_entry = {
                'resume_file': pdf_file,
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
                'llm_error': 'Could not find top job'
            }
            results.append(result_entry)
            continue
        
        # Use scores from find_top_job_for_resume (already computed)
        # Calculate matching skills count
        resume_skills = top_job.get('resume_skills', [])
        job_skills = top_job.get('job_skills', [])
        matching_skills = set(resume_skills) & set(job_skills)
        
        # LLM evaluation
        llm_result = {
            'llm_match': None, 
            'llm_reasoning': None, 
            'llm_recommendations': None,
            'linkedin_keywords': None,
            'llm_error': None
        }
        if use_llm_evaluation and llm_model:
            status_text.text(f"Evaluating with LLM {idx + 1}/{len(pdf_files)}: {pdf_file}")
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
            output_dir = os.path.join(workspace_path, "Resume_testing")
        else:
            output_dir = os.path.join("workspace", "Resume_testing")
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"evaluation_results_{timestamp}.json"
        json_path = os.path.join(output_dir, json_filename)
        
        # Prepare JSON data
        json_data = {
            'evaluation_timestamp': timestamp,
            'total_resumes': len(results),
            'evaluation_mode': 'individual_top_job_per_resume',
            'description': 'Each resume is matched with its top matching job from database (same logic as Resume Matching page)',
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
    st.success(f"✅ Evaluation complete! Processed {len(results)} resumes.")

# Display results
if st.session_state.get("evaluation_results"):
    st.markdown("---")
    st.markdown("### 📊 Evaluation Results")
    
    results = st.session_state.evaluation_results
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by final score descending
    df = df.sort_values('final_score', ascending=False)
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Resumes", len(df))
    with col2:
        avg_final = df['final_score'].mean()
        st.metric("Avg Final Score", f"{avg_final:.3f}")
    with col3:
        max_final = df['final_score'].max()
        st.metric("Best Match", f"{max_final:.3f}")
    with col4:
        min_final = df['final_score'].min()
        st.metric("Worst Match", f"{min_final:.3f}")
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
    display_df['Final Score %'] = (display_df['final_score'] * 100).round(2)
    display_df['Skill Score %'] = (display_df['skill_score'] * 100).round(2)
    display_df['Semantic Score %'] = (display_df['semantic_score'] * 100).round(2)
    display_df['Topic Score %'] = (display_df['topic_score'] * 100).round(2)
    
    # Format LLM match
    if 'llm_match' in display_df.columns:
        display_df['LLM Match'] = display_df['llm_match'].apply(
            lambda x: '✅ Yes' if x is True else '❌ No' if x is False else '⚠️ N/A'
        )
    
    # Select columns for display
    display_columns = [
        'resume_file', 'job_title', 'job_company', 'Final Score %', 'Skill Score %', 
        'Semantic Score %', 'Topic Score %',
        'resume_skills_count', 'job_skills_count', 'matching_skills_count'
    ]
    
    # Add LLM match column if available
    if 'LLM Match' in display_df.columns:
        display_columns.append('LLM Match')
    
    # Add rank column
    display_df['Rank'] = range(1, len(display_df) + 1)
    display_columns = ['Rank'] + display_columns
    
    st.dataframe(
        display_df[display_columns],
        use_container_width=True,
        height=600
    )
    
    # Expandable details for each resume
    st.markdown("#### Resume Details")
    
    for idx, row in df.iterrows():
        final_score_pct = row['final_score'] * 100
        
        # Color coding
        if final_score_pct >= 75:
            color = "🟢"
        elif final_score_pct >= 60:
            color = "🟡"
        else:
            color = "🟠"
        
        with st.expander(f"{color} {row['resume_file']} - {final_score_pct:.1f}% Match"):
            # Job information header
            if 'job_title' in row and row.get('job_title') != 'N/A':
                st.markdown(f"**Matched Job:** {row.get('job_title', 'N/A')} at {row.get('job_company', 'N/A')} (ID: {row.get('job_id', 'N/A')})")
                st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Scores:**")
                st.write(f"- Final Score: {final_score_pct:.2f}%")
                st.write(f"- Skill Score: {row['skill_score'] * 100:.2f}%")
                st.write(f"- Semantic Score: {row['semantic_score'] * 100:.2f}%")
                st.write(f"- Topic Score: {row['topic_score'] * 100:.2f}%")
                
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
            
            with col2:
                st.markdown("**Skills:**")
                st.write(f"- Resume Skills: {row['resume_skills_count']}")
                st.write(f"- Job Skills: {row['job_skills_count']}")
                st.write(f"- Matching Skills: {row['matching_skills_count']}")
            
            if row.get('error'):
                st.error(f"Error: {row['error']}")
    
    # Export results
    st.markdown("---")
    st.markdown("### 💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prepare export DataFrame for CSV
        export_df = df.copy()
        export_df['Final Score'] = export_df['final_score']
        export_df['Skill Score'] = export_df['skill_score']
        export_df['Semantic Score'] = export_df['semantic_score']
        export_df['Topic Score'] = export_df['topic_score']
        export_df['LLM Match'] = export_df['llm_match'].apply(
            lambda x: 'Yes' if x is True else 'No' if x is False else 'N/A'
        )
        
        # Format recommendations and keywords for CSV (convert lists to strings)
        export_df['LLM Recommendations'] = export_df['llm_recommendations'].apply(
            lambda x: '; '.join(x) if isinstance(x, list) and x else ('N/A' if x is None else str(x))
        )
        export_df['LinkedIn Keywords'] = export_df['linkedin_keywords'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) and x else ('N/A' if x is None else str(x))
        )
        export_df['LLM Reasoning'] = export_df['llm_reasoning'].apply(
            lambda x: str(x) if x else 'N/A'
        )
        
        export_df['Rank'] = range(1, len(export_df) + 1)
        
        export_columns = [
            'Rank', 'resume_file', 'job_title', 'job_company', 'job_id',
            'Final Score', 'Skill Score', 
            'Semantic Score', 'Topic Score', 'LLM Match', 'LLM Reasoning',
            'LLM Recommendations', 'LinkedIn Keywords',
            'resume_skills_count', 'job_skills_count', 'matching_skills_count',
            'resume_text_length'
        ]
        
        csv = export_df[export_columns].to_csv(index=False)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📥 Download Results as CSV",
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

# Footer
st.markdown("---")
st.markdown("""
### ℹ️ How It Works

1. **Resume Processing**: Extracts text from all PDF files in `workspace/Resume_testing/`
2. **Top Job Matching**: For each resume, finds its top matching job from the database using the same logic as Resume Matching page:
   - Generates SBERT embedding for the resume
   - Uses vector search to find similar jobs
   - Applies skill scoring and combined scoring
   - Returns the top matching job for that resume
3. **Evaluation**: For each resume-job pair, computes:
   - **Skill Score (NER)**: Jaccard similarity between resume and job skills
   - **Semantic Score**: Cosine similarity of SBERT embeddings
   - **Topic Score**: Currently uses semantic score as proxy
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
