"""
Vector Search and LLM Evaluation Functions
==========================================

This module provides all functions for vector search and evaluation with LLM
without any Streamlit dependencies. It can be used in notebooks, scripts, or
other applications.

Features:
- Text embedding generation (SBERT, Word2Vec)
- Vector similarity search
- Topic modeling integration
- Skill extraction and matching
- LLM-based resume evaluation
- Combined scoring functions

Author: Extracted from Streamlit app functions
Date: December 2025
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

# Optional imports with graceful degradation
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except Exception:
    PYPDF_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

try:
    import spacy
    from spacy.matcher import PhraseMatcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


# ============================================================================
# MASTER SKILL LIST
# ============================================================================

MASTER_SKILL_LIST = [
    # Technical / Data Skills
    "python", "r", "java", "javascript", "typescript",
    "c++", "c#", "scala", "go", "matlab",
    "bash", "shell scripting",
    "sql", "nosql", "postgresql", "mysql", "oracle", "sqlite",
    "mongodb", "snowflake", "redshift", "bigquery", "azure sql",
    "data analysis", "data analytics", "statistical analysis",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn",
    "plotly", "pyspark", "spark", "hadoop", "hive", "mapreduce",
    "machine learning", "deep learning", "neural networks",
    "logistic regression", "linear regression", "random forest",
    "xgboost", "lightgbm", "catboost",
    "svm", "knn", "decision trees", "pca", "kmeans",
    "gradient boosting", "model tuning", "feature engineering",
    "nlp", "natural language processing", "topic modeling",
    "lda", "lsa", "keyword extraction",
    "named entity recognition", "text classification",
    "sentiment analysis", "embeddings", "bert", "word2vec",
    "aws", "azure", "gcp", "docker", "kubernetes",
    "lambda", "ec2", "s3", "athena", "dynamodb",
    "databricks", "airflow", "cloud functions",
    "tableau", "power bi", "metabase", "looker", "qlik",
    "data visualization", "dashboard development",
    "etl", "elt", "data pipeline", "data ingestion",
    "data cleaning", "data transformation", "data integration",
    "git", "github", "gitlab", "bitbucket",
    "ci/cd", "jenkins",
    "sap", "sap erp", "salesforce", "salesforce crm",
    "hubspot", "hubspot crm", "airtable", "jira", "confluence", "notion",
    # Business & Analytics Skills
    "business analysis", "requirements gathering",
    "market research", "competitive analysis",
    "financial analysis", "risk analysis", "cost analysis",
    "forecasting", "trend analysis", "variance analysis",
    "p&l management", "strategic planning",
    "business modeling", "stakeholder management",
    "reporting", "presentation development",
    "process improvement", "process optimization",
    "root cause analysis", "gap analysis",
    "workflow automation", "operational efficiency",
    "kpi analysis", "performance analysis",
    "customer segmentation", "persona development",
    "data-driven decision making",
    "problem solving", "insights synthesis",
    "client communication", "proposal writing",
    "project scoping", "roadmap planning",
    "change management", "cross-functional collaboration",
    # Marketing / Sales / RevOps Skills
    "crm management", "lead generation", "pipeline management",
    "sales operations", "sales strategy", "sales forecasting",
    "revenue operations", "revops", "gtm strategy",
    "go-to-market", "account management",
    "client success", "customer retention",
    "digital marketing", "content marketing",
    "seo", "sem", "ppc", "email marketing",
    "campaign optimization", "social media analytics",
    "marketing automation", "google analytics",
    "google ads", "mailchimp", "marketo",
    "outreach", "gong", "zoominfo",
    "validation rules", "crm integrations",
    "funnel analysis", "data stamping",
    # Product Skills
    "product management", "product analytics",
    "a/b testing", "experiment design",
    "feature prioritization", "user research", "ux research",
    "user stories", "agile", "scrum", "kanban",
    "roadmap development", "user journey mapping",
    "requirements documentation",
    "market sizing", "competitive positioning",
    # Finance & Operations Skills
    "fp&a", "financial modeling", "budgeting",
    "scenario analysis", "invoice processing",
    "billing operations", "revenue analysis",
    "cost optimization",
    "supply chain management", "inventory management",
    "logistics", "procurement", "vendor management",
    "operations management", "kpi reporting",
    # Soft Skills
    "communication", "leadership", "teamwork",
    "collaboration", "critical thinking", "problem solving",
    "adaptability", "time management",
    "presentation skills", "negotiation",
    "public speaking", "project management",
    "detail oriented", "strategic thinking",
    "multitasking", "analytical thinking",
    "decision making", "organization skills"
]

MASTER_SKILL_LIST = list(set(MASTER_SKILL_LIST))


# ============================================================================
# TEXT CLEANING AND PREPROCESSING
# ============================================================================

def clean_text(text: str) -> str:
    """
    Basic text cleaning used for resumes and job descriptions.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\+?\d[\d\-\s\(\)]{7,}\d", " ", text)
    text = re.sub(r"(https?:\/\/\S+|www\.\S+)", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"[•●▪■◆▶►▸⦿⦾]", "- ", text)
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)

    lines = [line.strip() for line in text.split("\n")]
    final_lines: List[str] = []
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


def extract_text_from_pdf(file_path: str) -> Optional[str]:
    """
    Extract raw text from a PDF file.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Extracted text or None if extraction fails
    """
    if not PYPDF_AVAILABLE:
        return None
    try:
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip()
    except Exception:
        return None


def simple_tokenize(text):
    """
    Simple tokenization for Word2Vec.
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of tokens
    """
    if pd.isna(text):
        return []
    return str(text).split()


# ============================================================================
# SPACY AND NER FUNCTIONS
# ============================================================================

# Global cache for spaCy model
_SPACY_MODEL = None

def load_spacy_model():
    """Load spaCy model (cached)"""
    global _SPACY_MODEL
    if _SPACY_MODEL is not None:
        return _SPACY_MODEL
        
    if SPACY_AVAILABLE:
        try:
            _SPACY_MODEL = spacy.load("en_core_web_sm")
            return _SPACY_MODEL
        except Exception:
            return None
    return None


def build_skill_ner(skill_list: List[str]):
    """
    Builds a spaCy PhraseMatcher for custom skill extraction.
    
    Args:
        skill_list: List of skills to match
        
    Returns:
        PhraseMatcher or None if spaCy not available
    """
    if not SPACY_AVAILABLE:
        return None
    nlp = load_spacy_model()
    if nlp is None:
        return None
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in skill_list]
    matcher.add("SKILL", patterns)
    return matcher


def extract_skill_entities(text: str, skill_matcher):
    """
    Extracts skill entities from text using the SKILL PhraseMatcher.
    
    Args:
        text: Text to extract skills from
        skill_matcher: PhraseMatcher instance
        
    Returns:
        Sorted list of unique skills found
    """
    if not SPACY_AVAILABLE or skill_matcher is None:
        return []
    nlp = load_spacy_model()
    if nlp is None:
        return []
    doc = nlp(text)
    matches = skill_matcher(doc)
    skills_found = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        skills_found.add(span.text.lower())
    return sorted(list(skills_found))


def extract_skills_keywords(text: str, skill_list: List[str]) -> List[str]:
    """
    Keyword-only skill extraction (case-insensitive, multi-word aware).
    Fallback when spaCy is not available.
    
    Args:
        text: Text to extract skills from
        skill_list: List of skills to search for
        
    Returns:
        Sorted list of skills found
    """
    if not text or not skill_list:
        return []
    text_lower = text.lower()
    found = set()
    for skill in skill_list:
        sk = skill.lower().strip()
        if not sk:
            continue
        if " " in sk:
            if sk in text_lower:
                found.add(sk)
        else:
            if re.search(r"\b" + re.escape(sk) + r"\b", text_lower):
                found.add(sk)
    return sorted(found)


def skill_jaccard_score(resume_skills: List[str], job_skills: List[str]) -> float:
    """
    Jaccard similarity between resume skills and job skills.
    = |intersection| / |union|
    
    Args:
        resume_skills: List of resume skills
        job_skills: List of job skills
        
    Returns:
        Jaccard similarity score (0.0 to 1.0)
    """
    resume_set = set(resume_skills)
    job_set = set(job_skills)

    union = resume_set | job_set
    if not union:
        return 0.0

    overlap = resume_set & job_set
    score = len(overlap) / len(union)
    return score


# ============================================================================
# SBERT MODEL FUNCTIONS
# ============================================================================

# Global cache for SBERT model
_SBERT_MODEL = None

def load_sbert_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load pre-trained SBERT model from HuggingFace (cached).
    
    Args:
        model_name: Name of the pre-trained SBERT model to load. 
                   Default: "sentence-transformers/all-MiniLM-L6-v2" (384 dimensions)
                   Other options: 
                   - "sentence-transformers/all-mpnet-base-v2" (768 dimensions)
                   - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (384 dimensions, multilingual)
    
    Returns:
        SentenceTransformer model or None if loading fails
    """
    global _SBERT_MODEL
    if _SBERT_MODEL is not None:
        return _SBERT_MODEL
        
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None

    # Check for available devices in order of preference: CUDA, MPS, CPU
    device = "cpu"
    
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA GPU for SBERT model: {model_name}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Test MPS with a small tensor to ensure it works
        try:
            test_device = torch.device('mps')
            test_tensor = torch.randn(1).to(test_device)
            device = "mps"
            print(f"Using Apple MPS (Metal) for SBERT model: {model_name}")
        except Exception as e:
            print(f"MPS test failed ({e}), falling back to CPU")
            device = "cpu"
    else:
        device = "cpu"
        print(f"Using CPU for SBERT model: {model_name}")

    try:
        _SBERT_MODEL = SentenceTransformer(model_name, device=device)
        print(f"✅ Successfully loaded pre-trained SBERT model: {model_name}")
        return _SBERT_MODEL
    except Exception as e:
        print(f"Failed to load model {model_name} on {device}: {e}")
        # Fallback to CPU
        try:
            _SBERT_MODEL = SentenceTransformer(model_name, device="cpu")
            print(f"✅ Successfully loaded pre-trained SBERT model on CPU: {model_name}")
            return _SBERT_MODEL
        except Exception as e2:
            print(f"Failed to load model on CPU: {e2}")
            return None


def compute_job_embeddings_sbert(job_texts: List[str], model=None):
    """
    Compute embeddings for jobs using SBERT.
    
    Args:
        job_texts: List of job description texts
        model: SBERT model (if None, will load default)
        
    Returns:
        Numpy array of embeddings or None if computation fails
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
        
    if model is None:
        model = load_sbert_model()
        
    if model is None:
        return None

    # Ensure model is on the correct device
    device = next(model.parameters()).device
    print(f"Computing SBERT embeddings on device: {device}")

    # Adjust batch size based on device
    if device.type == 'mps':
        batch_size = 64  # Smaller for MPS memory
    elif device.type == 'cuda':
        batch_size = 128
    else:
        batch_size = 32  # Smaller for CPU memory

    try:
        embeddings = model.encode(
            job_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=False,
            device=device
        )
        return np.array(embeddings)
    except Exception as e:
        print(f"Embedding computation failed on {device}: {e}")
        # Try on CPU as fallback
        if device.type != 'cpu':
            print("Falling back to CPU...")
            try:
                cpu_model = model.to('cpu')
                embeddings = cpu_model.encode(
                    job_texts,
                    batch_size=16,  # Very small batch for CPU
                    show_progress_bar=True,
                    convert_to_tensor=False,
                    device='cpu'
                )
                return np.array(embeddings)
            except Exception as e2:
                print(f"CPU fallback also failed: {e2}")
                return None
        return None


# ============================================================================
# WORD2VEC MODEL FUNCTIONS
# ============================================================================

# Global cache for Word2Vec model
_WORD2VEC_MODEL = None

def load_trained_word2vec_model(
    use_pretrained: bool = False, 
    pretrained_path: str = None,
    models_dir: str = None
):
    """
    Load Word2Vec model - either a locally trained model or a pre-trained model (cached).
    
    Args:
        use_pretrained: If True, try to load a pre-trained Word2Vec model (e.g., Google's Word2Vec)
        pretrained_path: Path to pre-trained Word2Vec model file (e.g., GoogleNews-vectors-negative300.bin)
                       If None and use_pretrained=True, will look in models directory
        models_dir: Directory containing models (defaults to workspace/models)
    
    Returns:
        Word2Vec model or None if loading fails
    """
    global _WORD2VEC_MODEL
    if _WORD2VEC_MODEL is not None:
        return _WORD2VEC_MODEL
        
    if not GENSIM_AVAILABLE:
        return None

    if models_dir is None:
        # Default to workspace/models
        models_dir = os.path.join(os.getcwd(), "models")

    # Try to load pre-trained model first if requested
    if use_pretrained:
        # Common pre-trained model filenames
        pretrained_filenames = [
            "GoogleNews-vectors-negative300.bin",
            "GoogleNews-vectors-negative300.bin.gz",
            "word2vec-google-news-300.bin",
            "word2vec-google-news-300.bin.gz",
        ]
        
        # If specific path provided, use it
        if pretrained_path:
            pretrained_candidates = [pretrained_path]
        else:
            # Look for pre-trained models in models directory
            pretrained_candidates = [os.path.join(models_dir, fname) for fname in pretrained_filenames]
            # Also check current directory
            pretrained_candidates.extend([fname for fname in pretrained_filenames if os.path.exists(fname)])
        
        for pretrained_path_candidate in pretrained_candidates:
            if os.path.exists(pretrained_path_candidate):
                try:
                    print(f"Loading pre-trained Word2Vec model from {pretrained_path_candidate}...")
                    # Load pre-trained Word2Vec model (binary format)
                    _WORD2VEC_MODEL = Word2Vec.load_word2vec_format(pretrained_path_candidate, binary=True)
                    print(f"✅ Successfully loaded pre-trained Word2Vec model from {pretrained_path_candidate}")
                    print(f"   Vocabulary size: {len(_WORD2VEC_MODEL.wv.key_to_index):,} words")
                    print(f"   Vector size: {_WORD2VEC_MODEL.vector_size} dimensions")
                    return _WORD2VEC_MODEL
                except Exception as e:
                    print(f"Error loading pre-trained model from {pretrained_path_candidate}: {e}")
                    # Try as text format if binary fails
                    try:
                        _WORD2VEC_MODEL = Word2Vec.load_word2vec_format(pretrained_path_candidate, binary=False)
                        print(f"✅ Successfully loaded pre-trained Word2Vec model (text format) from {pretrained_path_candidate}")
                        print(f"   Vocabulary size: {len(_WORD2VEC_MODEL.wv.key_to_index):,} words")
                        print(f"   Vector size: {_WORD2VEC_MODEL.vector_size} dimensions")
                        return _WORD2VEC_MODEL
                    except Exception as e2:
                        print(f"Error loading as text format: {e2}")
                        continue

    # Try to load locally trained model
    if not JOBLIB_AVAILABLE:
        return None

    # Try different model filenames
    model_filenames = [
        "word2vec_model.joblib",
        "job_embeddings_w2v_14760_jobs_metadata.joblib",
        "w2v_model.joblib",
    ]
    
    for model_filename in model_filenames:
        model_path = os.path.join(models_dir, model_filename)
        
        if os.path.exists(model_path):
            try:
                model_data = joblib.load(model_path)
                # Handle different save formats
                if isinstance(model_data, dict):
                    if 'model' in model_data:
                        model = model_data['model']
                    else:
                        # Try to find Word2Vec model in dict
                        for key, value in model_data.items():
                            if value is not None and (hasattr(value, 'wv') or hasattr(value, 'vector_size')):
                                model = value
                                break
                        else:
                            continue
                else:
                    # Assume the saved data is the model itself
                    if model_data is not None and (hasattr(model_data, 'wv') or hasattr(model_data, 'vector_size')):
                        model = model_data
                    else:
                        continue
                
                # Validate model has required attributes
                if not (hasattr(model, 'wv') or hasattr(model, 'vector_size')):
                    continue
                
                _WORD2VEC_MODEL = model
                print(f"✅ Word2Vec model loaded from {model_path}")
                if hasattr(model, 'vector_size'):
                    print(f"   Vector size: {model.vector_size} dimensions")
                if hasattr(model, 'wv') and hasattr(model.wv, 'key_to_index'):
                    print(f"   Vocabulary size: {len(model.wv.key_to_index):,} words")
                return _WORD2VEC_MODEL
            except Exception as e:
                print(f"Error loading Word2Vec model from {model_path}: {e}")
                continue

    print(f"Word2Vec model not found in {models_dir}")
    return None


def train_word2vec_model(
    job_texts: List[str],
    resume_texts: List[str] = None,
    models_dir: str = None,
    save_model: bool = True
):
    """
    Train Word2Vec model on combined job and resume texts.
    
    Args:
        job_texts: List of job description texts
        resume_texts: Optional list of resume texts
        models_dir: Directory to save model (defaults to workspace/models)
        save_model: Whether to save the trained model
        
    Returns:
        Trained Word2Vec model
    """
    global _WORD2VEC_MODEL
    
    if not GENSIM_AVAILABLE:
        return None

    if models_dir is None:
        models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)

    model_filename = "word2vec_model.joblib"
    model_path = os.path.join(models_dir, model_filename)

    # Check if model already exists
    if os.path.exists(model_path) and JOBLIB_AVAILABLE:
        print(f"Loading existing Word2Vec model from {model_path}")
        try:
            model_data = joblib.load(model_path)
            _WORD2VEC_MODEL = model_data['model']
            return _WORD2VEC_MODEL
        except Exception as e:
            print(f"Error loading model: {e}")

    # Combine and tokenize
    all_texts = job_texts if resume_texts is None else job_texts + resume_texts
    training_corpus = [simple_tokenize(text) for text in all_texts if isinstance(text, str)]

    # Train model
    model = Word2Vec(
        sentences=training_corpus,
        vector_size=300,  # Standard Word2Vec dimension (300)
        window=5,
        min_count=10,
        workers=4,
        sg=1,
        epochs=10
    )

    # Save model if requested
    if save_model and JOBLIB_AVAILABLE:
        model_data = {
            'model': model,
            'trained_at': datetime.now().isoformat(),
            'vector_size': 300,
            'window': 5,
            'min_count': 10,
            'sg': 1,
            'epochs': 10
        }
        joblib.dump(model_data, model_path)
        print(f"✅ Word2Vec model saved to {model_path}")

    _WORD2VEC_MODEL = model
    return model


def get_doc_embedding_w2v(tokens: List[str], model):
    """
    Get document embedding using Word2Vec by averaging word vectors.
    
    Args:
        tokens: List of word tokens
        model: Word2Vec model
        
    Returns:
        Numpy array representing document embedding
    """
    if not GENSIM_AVAILABLE or model is None:
        # Get vector size from model if available, otherwise default to 300
        vector_size = getattr(model, 'vector_size', 300) if model else 300
        return np.zeros(vector_size, dtype="float32")

    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        vector_size = getattr(model, 'vector_size', 300)
        return np.zeros(vector_size, dtype="float32")
    return np.mean(vectors, axis=0)


def compute_job_embeddings_w2v(job_texts: List[str], model=None):
    """
    Compute embeddings for jobs using Word2Vec.
    
    Args:
        job_texts: List of job description texts
        model: Word2Vec model (if None, will load default)
        
    Returns:
        Numpy array of embeddings
    """
    if model is None:
        model = load_trained_word2vec_model()
        
    if model is None:
        return None
        
    embeddings = []
    for text in job_texts:
        tokens = simple_tokenize(text)
        emb = get_doc_embedding_w2v(tokens, model)
        embeddings.append(emb)
    return np.array(embeddings)


# ============================================================================
# TOPIC MODELING FUNCTIONS
# ============================================================================

# Global cache for topic models
_TOPIC_MODELS = {}

def load_trained_topic_model(method: str = 'LSA', n_topics: int = 100, models_dir: str = None):
    """
    Load trained topic model from disk (cached).
    
    Args:
        method: 'LDA' or 'LSA'
        n_topics: Number of topics
        models_dir: Directory containing models (defaults to workspace/models)
        
    Returns:
        Dictionary with 'vectorizer', 'model', and 'results' keys or None
    """
    global _TOPIC_MODELS
    
    cache_key = f"{method}_{n_topics}"
    if cache_key in _TOPIC_MODELS:
        return _TOPIC_MODELS[cache_key]
        
    if not JOBLIB_AVAILABLE:
        return None

    if models_dir is None:
        models_dir = os.path.join(os.getcwd(), "models")

    model_filename = f"topic_model_{method.lower()}_{n_topics}topics.joblib"
    model_path = os.path.join(models_dir, model_filename)

    if os.path.exists(model_path):
        try:
            model_data = joblib.load(model_path)
            result = {
                'vectorizer': model_data['vectorizer'],
                'model': model_data['model'],
                'results': model_data.get('results', None)
            }
            _TOPIC_MODELS[cache_key] = result
            print(f"✅ Topic model ({method}, {n_topics} topics) loaded from {model_path}")
            return result
        except Exception as e:
            print(f"Error loading topic model: {e}")
            return None
    else:
        print(f"Topic model not found at {model_path}")
        return None


def get_lsa_100_topics_model(models_dir: str = None):
    """
    Convenience function to get the LSA 100-topic model (most commonly used).
    
    Args:
        models_dir: Directory containing models
        
    Returns:
        Topic model data or None
    """
    return load_trained_topic_model(method='LSA', n_topics=100, models_dir=models_dir)


def get_document_topics(text: str, topic_model_data: Dict):
    """
    Get topic distribution for a document.
    
    Args:
        text: Document text
        topic_model_data: Dictionary with 'vectorizer' and 'model' keys
        
    Returns:
        Numpy array of topic distribution or None
    """
    if topic_model_data is None or not SKLEARN_AVAILABLE:
        return None

    vectorizer = topic_model_data['vectorizer']
    model = topic_model_data['model']

    # Transform the text
    dtm = vectorizer.transform([text])

    # Get topic distribution
    if hasattr(model, 'transform'):  # For LSA/SVD or LDA
        topic_dist = model.transform(dtm)[0]
    else:
        return None

    return topic_dist


def compute_topic_similarity(topic_dist1: np.ndarray, topic_dist2: np.ndarray) -> float:
    """
    Compute similarity between two topic distributions using cosine similarity.
    
    Args:
        topic_dist1: First topic distribution
        topic_dist2: Second topic distribution
        
    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    if topic_dist1 is None or topic_dist2 is None:
        return 0.0

    if not SKLEARN_AVAILABLE:
        return 0.0

    # Use cosine similarity for topic distributions
    dist1 = np.array(topic_dist1).reshape(1, -1)
    dist2 = np.array(topic_dist2).reshape(1, -1)

    similarity = cosine_similarity(dist1, dist2)[0][0]
    return similarity


def compute_topic_score(resume_text: str, job_text: str, topic_model_data=None) -> float:
    """
    Compute topic similarity using cached LSA model when available.
    
    Args:
        resume_text: Resume text
        job_text: Job description text
        topic_model_data: Topic model data (if None, will load default LSA 100 topics)
        
    Returns:
        Topic similarity score (0.0 to 1.0)
    """
    if topic_model_data is None:
        topic_model_data = get_lsa_100_topics_model()
        
    if topic_model_data is None:
        return 0.0
        
    try:
        r_topics = get_document_topics(resume_text, topic_model_data)
        j_topics = get_document_topics(job_text, topic_model_data)
        if r_topics is None or j_topics is None:
            return 0.0
        score = compute_topic_similarity(r_topics, j_topics)
        return float(np.clip(score, 0.0, 1.0))
    except Exception:
        return 0.0


# ============================================================================
# EMBEDDING GENERATION (UNIFIED INTERFACE)
# ============================================================================

def generate_embedding(text: str, method: str = "sbert", models_dir: str = None) -> Optional[np.ndarray]:
    """
    Generate embedding using specified method (unified interface).
    
    Args:
        text: Text to embed
        method: Either "sbert" or "word2vec"
        models_dir: Directory containing models (for word2vec)
        
    Returns:
        Numpy array embedding or None if generation fails
    """
    if method == "sbert":
        model = load_sbert_model()
        if model is None:
            return None
        device = next(model.parameters()).device
        try:
            return model.encode([text], batch_size=1, show_progress_bar=False, convert_to_tensor=False, device=device)[0]
        except Exception as e:
            print(f"SBERT encoding failed on {device}: {e}")
            # Fallback to CPU
            if device.type != 'cpu':
                try:
                    cpu_model = model.to('cpu')
                    return cpu_model.encode([text], batch_size=1, show_progress_bar=False, convert_to_tensor=False, device='cpu')[0]
                except Exception as e2:
                    print(f"CPU fallback failed: {e2}")
                    return None
            return None
            
    elif method == "word2vec" or method == "w2v":
        try:
            model = load_trained_word2vec_model(models_dir=models_dir)
            if model is None:
                print("Word2Vec model not available in generate_embedding()")
                return None
            tokens = simple_tokenize(text)
            emb = get_doc_embedding_w2v(tokens, model)
            return emb
        except Exception as e:
            print(f"Error generating Word2Vec embedding: {e}")
            return None
    else:
        print(f"Unknown embedding method: {method}")
        return None


# ============================================================================
# VECTOR SIMILARITY SEARCH
# ============================================================================

def find_similar_jobs_local(
    query_embedding: np.ndarray,
    job_embeddings: np.ndarray,
    df: pd.DataFrame,
    top_k: int = 10
) -> List[Dict]:
    """
    Find similar jobs using local embeddings and cosine similarity.
    
    Args:
        query_embedding: Query embedding vector
        job_embeddings: Matrix of job embeddings
        df: DataFrame containing job data
        top_k: Number of top matches to return
        
    Returns:
        List of dictionaries with job matches and similarity scores
    """
    if not SKLEARN_AVAILABLE or job_embeddings is None or query_embedding is None:
        return []

    query_emb = query_embedding.reshape(1, -1)
    similarities = cosine_similarity(query_emb, job_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        job_row = df.iloc[idx]
        results.append({
            'id': job_row.get('id', 'N/A'),
            'title': job_row.get('title', job_row.get('Job Title', 'N/A')),
            'company': job_row.get('company', job_row.get('Company', 'N/A')),
            'text': job_row.get('text', job_row.get('Description', 'N/A')),
            'similarity': similarities[idx]
        })
    return results


def find_similar_jobs_combined(
    query_text: str,
    job_texts: List[str],
    df: pd.DataFrame,
    top_k: int = 10,
    weights: Dict[str, float] = None,
    skill_list: List[str] = None,
    models_dir: str = None
) -> List[Dict]:
    """
    Find similar jobs using combined scoring (Word2Vec + Topic Modeling + Skills).
    
    Args:
        query_text: Resume or query text to match against
        job_texts: List of job description texts
        df: DataFrame containing job data
        top_k: Number of top matches to return
        weights: Weights for different similarity components
                 Default: {'skills': 0.45, 'semantic': 0.35, 'topic': 0.20}
        skill_list: List of skills to match (defaults to MASTER_SKILL_LIST)
        models_dir: Directory containing models
        
    Returns:
        List of dictionaries with job matches and combined similarity scores
    """
    if weights is None:
        weights = {'skills': 0.45, 'semantic': 0.35, 'topic': 0.20}
        
    if skill_list is None:
        skill_list = MASTER_SKILL_LIST

    # Load models
    w2v_model = load_trained_word2vec_model(models_dir=models_dir)
    topic_model_data = load_trained_topic_model(models_dir=models_dir)
    skill_matcher = build_skill_ner(skill_list)

    # Extract skills from query
    if skill_matcher:
        query_skills = extract_skill_entities(query_text, skill_matcher)
    else:
        query_skills = extract_skills_keywords(query_text, skill_list)

    # Get query embeddings and topics
    query_w2v_emb = None
    query_topics = None

    if w2v_model:
        query_tokens = simple_tokenize(query_text)
        query_w2v_emb = get_doc_embedding_w2v(query_tokens, w2v_model)

    if topic_model_data:
        query_topics = get_document_topics(query_text, topic_model_data)

    # Process each job
    similarities = []

    for idx, job_text in enumerate(job_texts):
        job_row = df.iloc[idx]

        # Skills similarity
        if skill_matcher:
            job_skills = extract_skill_entities(job_text, skill_matcher)
        else:
            job_skills = extract_skills_keywords(job_text, skill_list)
        skills_sim = skill_jaccard_score(query_skills, job_skills)

        # Semantic similarity (Word2Vec)
        semantic_sim = 0.0
        if w2v_model and query_w2v_emb is not None:
            job_tokens = simple_tokenize(job_text)
            job_w2v_emb = get_doc_embedding_w2v(job_tokens, w2v_model)
            if job_w2v_emb is not None:
                semantic_sim = cosine_similarity(
                    query_w2v_emb.reshape(1, -1),
                    job_w2v_emb.reshape(1, -1)
                )[0][0]

        # Topic similarity
        topic_sim = 0.0
        if topic_model_data and query_topics is not None:
            job_topics = get_document_topics(job_text, topic_model_data)
            if job_topics is not None:
                topic_sim = compute_topic_similarity(query_topics, job_topics)

        # Combined similarity score
        combined_sim = (
            weights['skills'] * skills_sim +
            weights['semantic'] * semantic_sim +
            weights['topic'] * topic_sim
        )

        similarities.append({
            'index': idx,
            'skills_sim': skills_sim,
            'semantic_sim': semantic_sim,
            'topic_sim': topic_sim,
            'combined_sim': combined_sim,
            'query_skills': query_skills,
            'job_skills': job_skills,
            'job_data': {
                'id': job_row.get('id', 'N/A'),
                'title': job_row.get('title', job_row.get('Job Title', 'N/A')),
                'company': job_row.get('company', job_row.get('Company', 'N/A')),
                'text': job_row.get('text', job_row.get('Description', 'N/A'))
            }
        })

    # Sort by combined similarity and return top_k
    similarities.sort(key=lambda x: x['combined_sim'], reverse=True)
    top_results = similarities[:top_k]

    # Format results
    results = []
    for sim in top_results:
        result = sim['job_data'].copy()
        result.update({
            'similarity': sim['combined_sim'],
            'skills_similarity': sim['skills_sim'],
            'semantic_similarity': sim['semantic_sim'],
            'topic_similarity': sim['topic_sim'],
            'query_skills': sim['query_skills'],
            'job_skills': sim['job_skills']
        })
        results.append(result)

    return results


# ============================================================================
# LLM EVALUATION WITH OLLAMA
# ============================================================================

def evaluate_with_llm(
    resume_text: str,
    job: Dict,
    model_name: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Dict:
    """
    LLM-based evaluation of resume against job using Ollama.
    Returns Yes/No match decision with reasoning and recommendations.
    
    Args:
        resume_text: Resume text
        job: Dictionary with job data (must have 'title', 'company', 'text' keys)
        model_name: Ollama model name (defaults to env vars or "gpt-oss:20b")
        api_url: Ollama API URL (defaults to env vars or "http://127.0.0.1:11434")
        
    Returns:
        Dictionary with:
        - llm_match: True/False/None
        - llm_reasoning: String explanation
        - llm_recommendations: List of improvement suggestions (if No match)
        - linkedin_keywords: List of keywords for LinkedIn search (if No match)
        - llm_error: Error message or None
    """
    if not OLLAMA_AVAILABLE:
        return {
            "llm_match": None,
            "llm_reasoning": None,
            "llm_recommendations": None,
            "linkedin_keywords": None,
            "llm_error": "Ollama not available",
        }
        
    model = model_name or os.getenv("OLLAMA_MODEL") or os.getenv("OLLAMA_DEFAULT_MODEL") or "gpt-oss:20b"
    host = api_url or os.getenv("OLLAMA_API_URL") or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434"
    
    try:
        client = ollama.Client(host=host)
    except Exception as e:
        return {
            "llm_match": None,
            "llm_reasoning": None,
            "llm_recommendations": None,
            "linkedin_keywords": None,
            "llm_error": str(e),
        }

    system_prompt = (
        "You are an expert recruiter. Focus on essential qualifications only. "
        "Say Yes if the candidate can reasonably do the job with their skills or transferable experience."
    )
    
    user_prompt = f"""Evaluate if this resume matches this job.

JOB TITLE: {job.get('title', 'N/A')}
COMPANY: {job.get('company', 'N/A')}
JOB DESCRIPTION:
{clean_text(job.get('text', ''))[:3000]}

RESUME:
{clean_text(resume_text)[:3000]}

Output exactly:
Line1: Yes or No
Line2: One-sentence reason
If No, Line3: RECOMMENDATIONS: - item1
- item2
- item3
If No, Line4: LINKEDIN_KEYWORDS: kw1, kw2, kw3"""

    try:
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = resp["message"]["content"].strip()
    except Exception as e:
        return {
            "llm_match": None,
            "llm_reasoning": None,
            "llm_recommendations": None,
            "linkedin_keywords": None,
            "llm_error": str(e),
        }

    # Parse response
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    match = None
    reason = None
    recs = None
    keywords = None

    if lines:
        first = lines[0].upper()
        if first.startswith("YES"):
            match = True
        elif first.startswith("NO"):
            match = False
        if len(lines) > 1:
            reason = lines[1]
        if match is False:
            for line in lines[2:]:
                upper = line.upper()
                if upper.startswith("RECOMMENDATIONS"):
                    recs = []
                    continue
                if upper.startswith("LINKEDIN_KEYWORDS"):
                    tail = line.split(":", 1)[1].strip() if ":" in line else ""
                    if tail:
                        keywords = [k.strip() for k in tail.split(",") if k.strip()]
                    break
                if recs is not None:
                    recs.append(line[1:].strip() if line.startswith("-") else line)

    return {
        "llm_match": match,
        "llm_reasoning": reason,
        "llm_recommendations": recs,
        "linkedin_keywords": keywords,
        "llm_error": None,
    }


# ============================================================================
# COMPLETE RESUME EVALUATION PIPELINE
# ============================================================================

def get_resume_and_job_skills(
    resume_text: str,
    job_text: str,
    skill_list: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Extract skills using matcher if available, else keyword fallback.
    
    Args:
        resume_text: Resume text
        job_text: Job description text
        skill_list: List of skills to match
        
    Returns:
        Tuple of (resume_skills, job_skills)
    """
    matcher = build_skill_ner(skill_list)

    if matcher:
        try:
            resume_skills = extract_skill_entities(resume_text, matcher)
            job_skills = extract_skill_entities(job_text, matcher)
        except Exception:
            resume_skills = extract_skills_keywords(resume_text, skill_list)
            job_skills = extract_skills_keywords(job_text, skill_list)
    else:
        resume_skills = extract_skills_keywords(resume_text, skill_list)
        job_skills = extract_skills_keywords(job_text, skill_list)
        
    return resume_skills, job_skills


def find_top_jobs_for_resume(
    resume_text: str,
    job_texts: List[str],
    df: pd.DataFrame,
    skill_list: List[str] = None,
    top_k: int = 3,
    embedding_type: str = "sbert",
    models_dir: str = None
) -> List[Dict]:
    """
    Vector search + multi-component scoring to find best job matches for resume.
    
    Args:
        resume_text: Resume text
        job_texts: List of job description texts
        df: DataFrame containing job data
        skill_list: List of skills to match (defaults to MASTER_SKILL_LIST)
        top_k: Number of top matches to return
        embedding_type: "sbert" or "word2vec"
        models_dir: Directory containing models
        
    Returns:
        List of dictionaries with enriched job matches
    """
    if skill_list is None:
        skill_list = MASTER_SKILL_LIST
        
    # Generate resume embedding
    resume_emb = generate_embedding(resume_text, method=embedding_type, models_dir=models_dir)
    if resume_emb is None:
        return []
        
    # Generate job embeddings
    if embedding_type == "sbert":
        job_embeddings = compute_job_embeddings_sbert(job_texts)
    else:
        job_embeddings = compute_job_embeddings_w2v(job_texts)
        
    if job_embeddings is None:
        return []
        
    # Get initial matches based on vector similarity
    matches = find_similar_jobs_local(
        resume_emb, job_embeddings, df, top_k=top_k * 2
    )
    
    if not matches:
        return []

    # Load topic model for additional scoring
    topic_model = get_lsa_100_topics_model(models_dir=models_dir)
    
    # Get resume skills
    resume_skills_kw = extract_skills_keywords(resume_text, skill_list)
    
    enriched = []
    for job in matches:
        job_text = job.get("text", "") or ""
        
        # Extract skills
        resume_skills, job_skills = get_resume_and_job_skills(
            resume_text, job_text, skill_list
        )
        if not resume_skills:
            resume_skills = resume_skills_kw
            
        # Compute different similarity scores
        skill_score = skill_jaccard_score(resume_skills, job_skills)
        semantic_score = job.get("similarity", 0.0)
        topic_score = compute_topic_score(resume_text, job_text, topic_model)
        
        if topic_score == 0.0:
            topic_score = semantic_score
            
        # Combined final score
        avg_ts = (topic_score + semantic_score) / 2
        final_score = avg_ts + (1 - avg_ts) * skill_score
        
        enriched.append(
            {
                **job,
                "skill_score": skill_score,
                "semantic_score": semantic_score,
                "topic_score": topic_score,
                "final_score": final_score,
                "resume_skills": resume_skills,
                "job_skills": job_skills,
            }
        )
        
    enriched.sort(key=lambda x: x["final_score"], reverse=True)
    return enriched[:top_k]


def evaluate_resume(
    resume_text: str,
    job_texts: List[str],
    df: pd.DataFrame,
    skill_list: List[str] = None,
    top_k: int = 3,
    embedding_type: str = "sbert",
    run_llm: bool = False,
    llm_model: Optional[str] = None,
    llm_api_url: Optional[str] = None,
    models_dir: str = None
) -> List[Dict]:
    """
    End-to-end resume evaluation pipeline.
    
    This is the main entry point for resume evaluation. It:
    1. Finds top matching jobs using vector search and combined scoring
    2. Optionally evaluates matches with LLM
    3. Returns comprehensive results with all scores and recommendations
    
    Args:
        resume_text: Resume text to evaluate
        job_texts: List of job description texts
        df: DataFrame containing job data
        skill_list: List of skills to match (defaults to MASTER_SKILL_LIST)
        top_k: Number of top matches to return
        embedding_type: "sbert" or "word2vec"
        run_llm: Whether to run LLM evaluation on matches
        llm_model: Ollama model name for LLM evaluation
        llm_api_url: Ollama API URL for LLM evaluation
        models_dir: Directory containing models
        
    Returns:
        List of dictionaries with complete evaluation results for each match
    """
    if skill_list is None:
        skill_list = MASTER_SKILL_LIST
        
    resume_text = clean_text(resume_text)
    
    jobs = find_top_jobs_for_resume(
        resume_text,
        job_texts=job_texts,
        df=df,
        skill_list=skill_list,
        top_k=top_k,
        embedding_type=embedding_type,
        models_dir=models_dir
    )
    
    results = []
    for rank, job in enumerate(jobs, start=1):
        matching_skills = set(job.get("resume_skills", [])) & set(job.get("job_skills", []))
        
        llm_result = {
            "llm_match": None,
            "llm_reasoning": None,
            "llm_recommendations": None,
            "linkedin_keywords": None,
            "llm_error": None,
        }
        
        if run_llm:
            llm_result = evaluate_with_llm(
                resume_text, job, model_name=llm_model, api_url=llm_api_url
            )
            
        results.append(
            {
                "job_rank": rank,
                "job_title": job.get("title", "N/A"),
                "job_company": job.get("company", "N/A"),
                "job_id": job.get("id", "N/A"),
                "skill_score": job.get("skill_score", 0.0),
                "semantic_score": job.get("semantic_score", 0.0),
                "topic_score": job.get("topic_score", 0.0),
                "final_score": job.get("final_score", 0.0),
                "resume_skills": job.get("resume_skills", []),
                "job_skills": job.get("job_skills", []),
                "matching_skills": sorted(matching_skills),
                **llm_result,
            }
        )
        
    return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_dependencies() -> Dict[str, bool]:
    """
    Check which dependencies are available.
    
    Returns:
        Dictionary mapping dependency name to availability boolean
    """
    return {
        'pypdf': PYPDF_AVAILABLE,
        'ollama': OLLAMA_AVAILABLE,
        'spacy': SPACY_AVAILABLE,
        'sklearn': SKLEARN_AVAILABLE,
        'gensim': GENSIM_AVAILABLE,
        'sentence_transformers': SENTENCE_TRANSFORMERS_AVAILABLE,
        'joblib': JOBLIB_AVAILABLE,
    }


def print_dependencies():
    """Print status of all dependencies."""
    deps = check_dependencies()
    print("Dependency Status:")
    print("-" * 40)
    for name, available in sorted(deps.items()):
        status = "✅ Available" if available else "❌ Not Available"
        print(f"{name:25s}: {status}")


if __name__ == "__main__":
    # When run as a script, print dependency status
    print_dependencies()

