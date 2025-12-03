import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import unicodedata
from datetime import datetime

from functions.nlp_config import TOPIC_MODEL_STOPWORDS, MASTER_SKILL_LIST
from components.header import load_css, render_header

# Add joblib import for model saving
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from utils import initialize_workspace

# Initialize workspace path and imports
initialize_workspace()

# Ensure theme and header are applied early
load_css()
# Try to import NLP libraries
try:
    import spacy
    from spacy.matcher import PhraseMatcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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

# Master skill list from NER notebook (530+ skills)
MASTER_SKILL_LIST = [
    # Technical / Programming
    "python", "r", "java", "javascript", "typescript",
    "c++", "c#", "scala", "go", "matlab",
    "bash", "shell scripting",
    "software engineering", "software development",
    "full stack development", "frontend development", "backend development",
    "api design", "rest apis", "microservices",
    "distributed systems", "scalable systems",
    "cloud infrastructure", "cloud computing", "cloud native", "cloud platforms",

    # Data Analytics
    "sql", "nosql", "postgresql", "mysql", "oracle", "sqlite",
    "mongodb", "snowflake", "redshift", "bigquery", "azure sql",
    "data analysis", "data analytics", "statistical analysis",
    "business intelligence", "operational reporting",
    "process mapping", "requirements analysis",
    "risk management", "financial reporting",

    # Data Tools
    "pandas", "numpy", "scipy", "matplotlib", "seaborn",
    "plotly", "pyspark", "spark", "hadoop", "hive", "mapreduce", "jira",

    # Machine Learning
    "machine learning", "deep learning", "neural networks",
    "logistic regression", "linear regression", "random forest",
    "xgboost", "lightgbm", "catboost",
    "svm", "knn", "decision trees", "pca", "kmeans",
    "gradient boosting", "model tuning", "feature engineering",

    # NLP
    "nlp", "natural language processing", "topic modeling",
    "lda", "lsa", "keyword extraction",
    "named entity recognition", "text classification",
    "sentiment analysis", "embeddings", "bert", "word2vec",

    # Cloud
    "aws", "azure", "gcp", "docker", "kubernetes",
    "lambda", "ec2", "s3", "athena", "dynamodb",
    "databricks", "airflow", "cloud functions",

    # BI Tools
    "tableau", "power bi", "metabase", "looker", "qlik",
    "data visualization", "dashboard development",

    # ETL / Pipelines
    "etl", "elt", "data pipeline", "data ingestion",
    "data cleaning", "data transformation", "data integration",

    # Version Control & DevOps
    "git", "github", "gitlab", "bitbucket",
    "ci/cd", "jenkins",

    # Enterprise Tools
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

    # Consulting skills
    "problem solving", "insights synthesis",
    "client communication", "proposal writing",
    "project scoping", "roadmap planning",
    "change management", "cross-functional collaboration",

    # Marketing / Sales
    "crm management", "lead generation", "pipeline management",
    "sales operations", "sales strategy", "sales forecasting",
    "revenue operations", "revops", "gtm strategy",
    "go-to-market", "account management",
    "client success", "customer retention", "digital marketing",
    "content marketing", "seo", "sem", "ppc", "email marketing",
    "campaign optimization", "social media analytics",

    # Marketing tools
    "marketing automation", "google analytics",
    "google ads", "mailchimp", "marketo",
    "outreach", "gong", "zoominfo",

    # RevOps Processes
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

    # Operations & Supply Chain
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
    "decision making", "organization skills",
    "attention to detail", "stakeholder communication",
    "conflict resolution", "problem-solving skills",
    "relationship building", "coaching", "mentoring",
]

# Skill normalization dictionary - maps variations to canonical skill names
SKILL_NORMALIZATION = {
    "python programming": "python",
    "java development": "java",
    "java programming": "java",
    "javascript programming": "javascript",
    "js": "javascript",
    "sql database": "sql",
    "sql queries": "sql",
    "machine learning algorithms": "machine learning",
    "ml algorithms": "machine learning",
    "deep learning models": "deep learning",
    "neural network": "neural networks",
    "data science": "data analysis",
    "data analytics": "data analysis",
    "software engineering": "software development",
    "web development": "full stack development",
    "front-end": "frontend development",
    "front end": "frontend development",
    "back-end": "backend development",
    "back end": "backend development",
    "rest api": "rest apis",
    "api": "rest apis",
    "cloud services": "cloud computing",
    "aws cloud": "aws",
    "azure cloud": "azure",
    "gcp cloud": "gcp",
    "git version control": "git",
    "version control": "git",
    "ci cd": "ci/cd",
    "continuous integration": "ci/cd",
    "natural language processing": "nlp",
    "text analytics": "nlp",
    "business intelligence tools": "business intelligence",
    "bi tools": "business intelligence",
    "data visualization tools": "data visualization",
    "dashboard creation": "dashboard development",
    "etl pipelines": "etl",
    "data pipelines": "data pipeline",
    "project management tools": "project management",
    "pm": "project management",
    "agile methodology": "agile",
    "scrum framework": "scrum",
    "kanban board": "kanban",
}

# Abbreviation expansion dictionary - maps abbreviations to full terms
ABBREVIATIONS = {
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "api": "rest apis",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "bi": "business intelligence",
    "etl": "extract transform load",
    "elt": "extract load transform",
    "ci/cd": "continuous integration continuous deployment",
    "crm": "customer relationship management",
    "erp": "enterprise resource planning",
    "sql": "structured query language",
    "nosql": "not only sql",
    "ui": "user interface",
    "ux": "user experience",
    "api": "application programming interface",
    "sdk": "software development kit",
    "sas": "statistical analysis system",
    "spss": "statistical package for the social sciences",
    "fp&a": "financial planning and analysis",
    "p&l": "profit and loss",
    "kpi": "key performance indicator",
    "gtm": "go-to-market",
    "revops": "revenue operations",
    "seo": "search engine optimization",
    "sem": "search engine marketing",
    "ppc": "pay per click",
    "cpq": "configure price quote",
    "kyc": "know your customer",
    "aml": "anti-money laundering",
    "gdpr": "general data protection regulation",
    "sox": "sarbanes-oxley act",
    "hipaa": "health insurance portability and accountability act",
    "fda": "food and drug administration",
    "gcp": "good clinical practice",
    "emr": "electronic medical records",
    "ehr": "electronic health records",
    "pmp": "project management professional",
    "itil": "information technology infrastructure library",
}

# Skill categories for better organization and visualization
SKILL_CATEGORIES = {
    "Programming Languages": ["python", "r", "java", "javascript", "typescript", "c++", "c#", "scala", "go", "matlab", "bash", "shell scripting", "php", "ruby", "swift", "kotlin", "c", "perl", "rust", "haskell"],
    "Frameworks & Libraries": ["django", "flask", "fastapi", "react", "react native", "angular", "vue.js", "next.js", "node.js", "express.js", "ruby on rails", "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly"],
    "Databases": ["sql", "nosql", "postgresql", "mysql", "oracle", "sqlite", "mongodb", "snowflake", "redshift", "bigquery", "azure sql", "sql server", "db2"],
    "Cloud & DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "lambda", "ec2", "s3", "athena", "dynamodb", "databricks", "airflow", "terraform", "ansible", "ci/cd", "jenkins", "github actions"],
    "Machine Learning": ["machine learning", "deep learning", "neural networks", "logistic regression", "linear regression", "random forest", "xgboost", "lightgbm", "catboost", "svm", "knn", "decision trees", "pca", "kmeans", "gradient boosting"],
    "NLP & AI": ["nlp", "natural language processing", "topic modeling", "lda", "lsa", "keyword extraction", "named entity recognition", "text classification", "sentiment analysis", "embeddings", "bert", "word2vec"],
    "Data Tools": ["pyspark", "spark", "hadoop", "hive", "mapreduce", "excel", "microsoft excel", "google sheets", "sas", "stata", "spss", "power query", "power pivot"],
    "BI & Visualization": ["tableau", "power bi", "metabase", "looker", "qlik", "data visualization", "dashboard development", "mode analytics", "amplitude", "mixpanel"],
    "Enterprise Tools": ["sap", "salesforce", "hubspot", "airtable", "jira", "confluence", "notion", "workday", "quickbooks", "xero", "netsuite"],
    "Testing & QA": ["unit testing", "integration testing", "qa testing", "automation testing", "selenium", "cypress", "pytest", "junit"],
    "Security": ["network security", "firewall configuration", "penetration testing", "vulnerability assessment", "siem", "splunk", "ssl", "tls", "vpn"],
    "Business Skills": ["business analysis", "requirements gathering", "market research", "competitive analysis", "financial analysis", "risk analysis", "forecasting", "strategic planning", "stakeholder management"],
    "Product & Design": ["product management", "product analytics", "a/b testing", "user research", "ux research", "figma", "sketch", "adobe xd", "wireframing", "prototyping"],
    "Marketing & Sales": ["crm management", "lead generation", "sales operations", "digital marketing", "content marketing", "seo", "sem", "ppc", "email marketing", "google analytics", "marketing automation"],
    "Soft Skills": ["communication", "leadership", "teamwork", "collaboration", "critical thinking", "problem solving", "adaptability", "time management", "presentation skills", "project management"],
}

# Context words that indicate skill mentions are valid
SKILL_CONTEXT_WORDS = [
    "experience", "proficient", "knowledge", "skills", "familiar", "expert", "expertise",
    "proficiency", "competent", "skilled", "experienced", "background", "qualifications",
    "requirements", "must have", "should have", "nice to have", "preferred", "required",
    "working knowledge", "hands-on", "practical", "demonstrated", "proven", "strong",
    "solid", "extensive", "deep", "comprehensive", "advanced", "intermediate", "beginner"
]

def normalize_skill(skill):
    """
    Normalize skill name to canonical form.
    
    Args:
        skill: Skill name string (lowercase)
    
    Returns:
        Normalized skill name
    """
    skill_lower = skill.lower().strip()
    return SKILL_NORMALIZATION.get(skill_lower, skill_lower)

def expand_abbreviations(skills):
    """
    Expand abbreviations in skill list to include full terms.
    
    Args:
        skills: List of skill names
    
    Returns:
        List of skills with abbreviations expanded
    """
    expanded = set()
    for skill in skills:
        skill_lower = skill.lower()
        expanded.add(skill_lower)
        # Add abbreviation expansion if exists
        if skill_lower in ABBREVIATIONS:
            expanded.add(ABBREVIATIONS[skill_lower])
        # Also check if any skill is an abbreviation for this one
        for abbrev, full_term in ABBREVIATIONS.items():
            if skill_lower == full_term.lower():
                expanded.add(abbrev)
    return list(expanded)

def categorize_skill(skill):
    """
    Categorize a skill into its primary category.
    
    Args:
        skill: Skill name string
    
    Returns:
        Category name or "Other"
    """
    skill_lower = skill.lower()
    for category, skills in SKILL_CATEGORIES.items():
        if skill_lower in [s.lower() for s in skills]:
            return category
    return "Other"

def is_valid_skill_context(doc, start, end, window=10):
    """
    Check if skill match is in valid context (near skill-related keywords).
    
    Args:
        doc: spaCy Doc object
        start: Start token index
        end: End token index
        window: Number of tokens to check before/after
    
    Returns:
        True if context suggests valid skill mention
    """
    # Get context window
    context_start = max(0, start - window)
    context_end = min(len(doc), end + window)
    context_text = doc[context_start:context_end].text.lower()
    
    # Check for skill-related context words
    return any(word in context_text for word in SKILL_CONTEXT_WORDS)

# NER functions with improvements
@st.cache_resource
def load_spacy_model(model_name="en_core_web_sm"):
    """
    Load spaCy model with model selection support.
    
    Cached per model name to avoid reloading the same model.
    
    Args:
        model_name: Name of spaCy model to load
            - "en_core_web_sm": Small English model (default, fast, ~50MB)
            - "en_core_web_md": Medium English model (better accuracy, ~40MB + vectors)
            - "en_core_web_lg": Large English model (best accuracy, slower, ~560MB + vectors)
    
    Returns:
        Loaded spaCy model or None if unavailable
    """
    if not SPACY_AVAILABLE:
        return None
    
    try:
        return spacy.load(model_name)
    except OSError:
        # Model not found, try to provide helpful error
        if model_name != "en_core_web_sm":
            # Try fallback to small model
            try:
                # Note: We can't show warnings in cached functions easily, 
                # so we'll handle this in the calling code
                return spacy.load("en_core_web_sm")
            except:
                return None
        return None
    except Exception as e:
        # Log error but don't show in cached function
        return None

@st.cache_resource
def build_skill_ner(skill_list, _nlp=None):
    """
    Builds a spaCy PhraseMatcher for custom skill extraction.
    
    Args:
        skill_list: List of skill names to match
        _nlp: spaCy model (for caching purposes)
    
    Returns:
        PhraseMatcher configured for skill extraction
    """
    if not SPACY_AVAILABLE:
        return None
    
    if _nlp is None:
        _nlp = load_spacy_model()
    if _nlp is None:
        return None
    
    # Use case-insensitive matching
    matcher = PhraseMatcher(_nlp.vocab, attr="LOWER")
    
    # Create patterns for each skill
    patterns = [_nlp.make_doc(skill) for skill in skill_list]
    matcher.add("SKILL", patterns)
    
    return matcher

def extract_skill_entities(text, skill_matcher, nlp=None, use_word_boundaries=True, 
                          use_overlap_resolution=True, use_context_filter=False):
    """
    Extract skill entities from text with improved matching.
    
    Improvements:
    1. Overlap resolution: Prefer longer matches over shorter ones
    2. Word boundary validation: Prevent partial matches (e.g., "java" in "javascript")
    3. Context filtering: Only keep skills mentioned in valid contexts
    4. Normalization: Map skill variations to canonical forms
    
    Args:
        text: Input text to extract skills from
        skill_matcher: PhraseMatcher for skills
        nlp: spaCy model (if None, loads default)
        use_word_boundaries: Check word boundaries to avoid partial matches
        use_overlap_resolution: Resolve overlapping matches (prefer longer)
        use_context_filter: Filter based on context words
    
    Returns:
        Sorted list of unique normalized skill names
    """
    if not SPACY_AVAILABLE or skill_matcher is None:
        return []
    
    if nlp is None:
        nlp = load_spacy_model()
    if nlp is None:
        return []
    
    doc = nlp(text)
    matches = skill_matcher(doc)
    
    if not matches:
        return []
    
    # Sort matches by length (longest first) for overlap resolution
    if use_overlap_resolution:
        matches = sorted(matches, key=lambda x: (x[2] - x[1]), reverse=True)
    
    skills_found = set()
    used_spans = set()
    
    for match_id, start, end in matches:
        # Overlap resolution: skip if overlaps with already used span
        if use_overlap_resolution:
            span_tuple = (start, end)
            # Check for overlap
            overlaps = False
            for used_start, used_end in used_spans:
                if not (end <= used_start or start >= used_end):
                    overlaps = True
                    break
            if overlaps:
                continue
        
        # Word boundary validation
        if use_word_boundaries:
            # Check if match is at word boundary
            if start > 0:
                prev_token = doc[start - 1]
                if prev_token.is_alpha or prev_token.is_digit:
                    # Check if it's part of a compound word (allow hyphens, underscores)
                    if not (prev_token.text in ['-', '_'] or doc[start].text in ['-', '_']):
                        continue
            
            if end < len(doc):
                next_token = doc[end] if end < len(doc) else None
                if next_token and (next_token.is_alpha or next_token.is_digit):
                    # Check if it's part of a compound word
                    if not (doc[end - 1].text in ['-', '_'] or next_token.text in ['-', '_']):
                        continue
        
        # Context filtering
        if use_context_filter:
            if not is_valid_skill_context(doc, start, end):
                continue
        
        # Extract and normalize skill
        span = doc[start:end]
        skill_text = span.text.lower().strip()
        normalized_skill = normalize_skill(skill_text)
        skills_found.add(normalized_skill)
        
        if use_overlap_resolution:
            used_spans.add((start, end))
    
    return sorted(list(skills_found))

def extract_skills_from_doc(doc, skill_matcher, use_word_boundaries=True, 
                            use_overlap_resolution=True, use_context_filter=False):
    """
    Extract skills directly from a spaCy Doc object (optimized version).
    
    This avoids re-processing the document.
    """
    matches = skill_matcher(doc)
    
    if not matches:
        return []
    
    # Sort matches by length (longest first) for overlap resolution
    if use_overlap_resolution:
        matches = sorted(matches, key=lambda x: (x[2] - x[1]), reverse=True)
    
    skills_found = set()
    used_spans = set()
    
    for match_id, start, end in matches:
        # Overlap resolution
        if use_overlap_resolution:
            overlaps = False
            for used_start, used_end in used_spans:
                if not (end <= used_start or start >= used_end):
                    overlaps = True
                    break
            if overlaps:
                continue
        
        # Word boundary validation
        if use_word_boundaries:
            if start > 0:
                prev_token = doc[start - 1]
                if prev_token.is_alpha or prev_token.is_digit:
                    if not (prev_token.text in ['-', '_'] or doc[start].text in ['-', '_']):
                        continue
            
            if end < len(doc):
                next_token = doc[end] if end < len(doc) else None
                if next_token and (next_token.is_alpha or next_token.is_digit):
                    if not (doc[end - 1].text in ['-', '_'] or next_token.text in ['-', '_']):
                        continue
        
        # Context filtering
        if use_context_filter:
            if not is_valid_skill_context(doc, start, end):
                continue
        
        # Extract and normalize skill
        span = doc[start:end]
        skill_text = span.text.lower().strip()
        normalized_skill = normalize_skill(skill_text)
        skills_found.add(normalized_skill)
        
        if use_overlap_resolution:
            used_spans.add((start, end))
    
    return sorted(list(skills_found))

def extract_skills_batch(
    texts,
    skill_matcher,
    nlp=None,
    batch_size=500,
    use_word_boundaries=True,
    use_overlap_resolution=True,
    use_context_filter=False,
    progress_callback=None,
    n_process=1,
):
    """
    Extract skills from multiple texts efficiently using optimized batch processing.
    
    Args:
        texts: List of text strings
        skill_matcher: PhraseMatcher for skills
        nlp: spaCy model
        batch_size: Number of texts to process at once (increased default for large datasets)
        use_word_boundaries: Enable word boundary checking
        use_overlap_resolution: Enable overlap resolution
        use_context_filter: Enable context filtering
        progress_callback: Optional callback function(processed, total) for progress updates
    
    Returns:
        List of skill lists (one per input text)
    """
    if nlp is None:
        nlp = load_spacy_model()
    if nlp is None:
        return [[] for _ in texts]
    
    all_results = []
    total_texts = len(texts)
    
    # Process in batches for efficiency (with optional parallel workers)
    for i in range(0, total_texts, batch_size):
        batch = texts[i:i+batch_size]
        # Use spaCy's pipe for efficient batch processing
        # Disable unnecessary components for speed (only need tokenizer and NER)
        pipe_kwargs = dict(batch_size=batch_size, disable=["parser", "lemmatizer"])
        try:
            # spaCy v3: supports n_process for multiprocessing
            docs = list(nlp.pipe(batch, n_process=max(1, int(n_process)), **pipe_kwargs))
        except TypeError:
            # spaCy v2 fallback: no n_process, but still batched
            docs = list(nlp.pipe(batch, **pipe_kwargs))
        
        # Extract skills directly from docs (no re-processing)
        for doc in docs:
            skills = extract_skills_from_doc(
                doc, skill_matcher,
                use_word_boundaries=use_word_boundaries,
                use_overlap_resolution=use_overlap_resolution,
                use_context_filter=use_context_filter
            )
            all_results.append(skills)
        
        # Progress callback
        if progress_callback:
            progress_callback(min(i + batch_size, total_texts), total_texts)
    
    return all_results

def extract_spacy_entities_batch(
    texts,
    nlp=None,
    batch_size=500,
    progress_callback=None,
    n_process=1,
):
    """
    Extract spaCy NER entities from multiple texts efficiently using batch processing.
    
    Args:
        texts: List of text strings
        nlp: spaCy model
        batch_size: Number of texts to process at once
        progress_callback: Optional callback function(processed, total) for progress updates
    
    Returns:
        List of entity lists (one per input text)
    """
    if nlp is None:
        nlp = load_spacy_model()
    if nlp is None:
        return [[] for _ in texts]
    
    all_results = []
    total_texts = len(texts)
    
    # Process in batches (with optional parallel workers)
    for i in range(0, total_texts, batch_size):
        batch = texts[i:i+batch_size]
        # Use spaCy's pipe with only NER component enabled
        pipe_kwargs = dict(batch_size=batch_size, disable=["parser", "lemmatizer", "tagger"])
        try:
            docs = list(nlp.pipe(batch, n_process=max(1, int(n_process)), **pipe_kwargs))
        except TypeError:
            docs = list(nlp.pipe(batch, **pipe_kwargs))
        
        for doc in docs:
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            all_results.append(entities)
        
        # Progress callback
        if progress_callback:
            progress_callback(min(i + batch_size, total_texts), total_texts)
    
    return all_results

def extract_spacy_entities(text, nlp=None):
    """
    Extract spaCy NER entities (ORG, GPE, DATE, PERSON, etc.)
    
    Args:
        text: Input text
        nlp: spaCy model (if None, loads default)
    
    Returns:
        List of (entity_text, entity_label) tuples
    """
    if not SPACY_AVAILABLE:
        return []
    if nlp is None:
        nlp = load_spacy_model()
    if nlp is None:
        return []
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def strip_experience(text):
    """Strip experience requirements from text"""
    if not isinstance(text, str):
        return ""
    import re
    return re.sub(r"Experience required:\s*\d+\s*(to\s*\d+)?\s*Years", "", text, flags=re.IGNORECASE).strip()

@st.cache_data
def run_topic_modeling(texts, method='LDA', n_topics=10, n_words=10, save_model=True, n_jobs=None):
    """Run topic modeling on texts.
    
    Args:
        texts: Iterable of job description strings.
        method: Either 'LDA' or 'LSA'.
        n_topics: Desired number of topics/components.
        n_words: Number of words to surface per topic.
        save_model: Whether to persist the trained model artifacts.
        n_jobs: Number of CPU workers for scikit-learn LDA (-1 uses all cores).
    """
    if not SKLEARN_AVAILABLE:
        return None
    
    # Try to import cuML for GPU acceleration
    try:
        from cuml.decomposition import LatentDirichletAllocation as cuLDA
        from cuml.decomposition import TruncatedSVD as cuSVD
        CUM_AVAILABLE = True
    except ImportError:
        CUM_AVAILABLE = False
    
    # Create models directory if it doesn't exist
    workspace_path = st.session_state.get('workspace_path')
    if workspace_path:
        models_dir = os.path.join(workspace_path, "models")
        os.makedirs(models_dir, exist_ok=True)
    else:
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
    
    model_filename = f"topic_model_{method.lower()}_{n_topics}topics.joblib"
    model_path = os.path.join(models_dir, model_filename)
    
    # Check if model already exists
    if os.path.exists(model_path) and JOBLIB_AVAILABLE:
        st.info(f"Loading existing {method} model from {model_path}")
        try:
            saved_data = joblib.load(model_path)
            # Return the saved results
            return saved_data['results']
        except Exception as e:
            st.warning(f"Could not load saved model: {e}. Retraining...")
    
    # Preprocess texts
    processed_texts = [strip_experience(text) for text in texts if isinstance(text, str)]
    
    if method == 'LDA':
        # LDA configuration
        vectorizer = CountVectorizer(
            strip_accents="unicode",
            stop_words=TOPIC_MODEL_STOPWORDS,
            lowercase=True,
            max_features=5000,
            token_pattern=r"\b[a-zA-Z]{3,}\b",
            max_df=0.75,
            min_df=5,
            ngram_range=(1, 3),
        )
        dtm = vectorizer.fit_transform(processed_texts)
        
        if CUM_AVAILABLE:
            st.info("Using GPU-accelerated cuML for LDA")
            lda = cuLDA(
                n_components=n_topics,
                max_iter=100,
                random_state=44,
            )
        else:
            effective_jobs = max(1, int(n_jobs)) if n_jobs else -1
            core_label = "all available CPU cores" if effective_jobs == -1 else f"{effective_jobs} CPU core{'s' if effective_jobs != 1 else ''}"
            st.info(f"Using CPU-based scikit-learn for LDA ({core_label})")
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=100,
                learning_method="batch",
                random_state=44,
                n_jobs=effective_jobs,
            )
        topics = lda.fit_transform(dtm)
        
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top words for each topic
        topic_words = []
        for topic_idx in range(n_topics):
            top_words_idx = lda.components_[topic_idx].argsort()[:-n_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_words.append(top_words)
            
        results = {
            'method': 'LDA',
            'n_topics': n_topics,
            'topic_words': topic_words,
            'vocab_size': len(feature_names)
        }
        
        # Save model if requested
        if save_model and JOBLIB_AVAILABLE:
            model_data = {
                'vectorizer': vectorizer,
                'model': lda,
                'results': results,
                'method': method,
                'n_topics': n_topics,
                'trained_at': datetime.now().isoformat()
            }
            joblib.dump(model_data, model_path)
            st.success(f"✅ LDA model saved to {model_path}")
        
        return results
    
    elif method == 'LSA':
        # LSA configuration
        vectorizer = TfidfVectorizer(
            stop_words=TOPIC_MODEL_STOPWORDS,
            lowercase=True,
            strip_accents="unicode",
            max_features=None,
            ngram_range=(1, 3),
            min_df=5,
            max_df=0.6,
            dtype=np.float32,
        )
        tfidf = vectorizer.fit_transform(processed_texts)
        
        if CUM_AVAILABLE:
            st.info("Using GPU-accelerated cuML for LSA")
            svd = cuSVD(n_components=min(n_topics, tfidf.shape[1]-1), random_state=42)
        else:
            st.info("Using CPU-based scikit-learn for LSA")
            svd = TruncatedSVD(n_components=min(n_topics, tfidf.shape[1]-1), random_state=42)
        topics = svd.fit_transform(tfidf)
        
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top words for each topic
        topic_words = []
        for topic_idx in range(min(n_topics, svd.components_.shape[0])):
            top_words_idx = svd.components_[topic_idx].argsort()[:-n_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_words.append(top_words)
            
        results = {
            'method': 'LSA',
            'n_topics': len(topic_words),
            'topic_words': topic_words,
            'explained_variance': svd.explained_variance_ratio_.sum()
        }
        
        # Save model if requested
        if save_model and JOBLIB_AVAILABLE:
            model_data = {
                'vectorizer': vectorizer,
                'model': svd,
                'results': results,
                'method': method,
                'n_topics': n_topics,
                'trained_at': datetime.now().isoformat()
            }
            joblib.dump(model_data, model_path)
            st.success(f"✅ LSA model saved to {model_path}")
        
        return results
    
    return None

@st.cache_data
def simple_tokenize(text):
    """Simple tokenization"""
    if pd.isna(text):
        return []
    return str(text).split()

@st.cache_resource
def train_word2vec_model(job_texts, resume_texts, save_model=True, workers=4):
    """Train Word2Vec model on combined job and resume texts"""
    if not GENSIM_AVAILABLE:
        return None
    
    # Create models directory if it doesn't exist
    workspace_path = st.session_state.get('workspace_path')
    if workspace_path:
        models_dir = os.path.join(workspace_path, "models")
        os.makedirs(models_dir, exist_ok=True)
    else:
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
    
    model_filename = "word2vec_model.joblib"
    model_path = os.path.join(models_dir, model_filename)
    
    # Combine and tokenize
    training_corpus = [simple_tokenize(text) for text in job_texts + resume_texts if isinstance(text, str)]
    
    # Train model
    model = Word2Vec(
        sentences=training_corpus,
        vector_size=300,
        window=5,
        min_count=10,
        workers=max(1, int(workers)),
        sg=1,
        epochs=10
    )
    
    # Save model if requested
    if save_model and JOBLIB_AVAILABLE:
        model_data = {
            'model': model,
            'trained_at': datetime.now().isoformat(),
            'vector_size': 300,  # Word2Vec dimension
            'window': 5,
            'min_count': 10,
            'sg': 1,
            'epochs': 10
        }
        joblib.dump(model_data, model_path)
        st.success(f"✅ Word2Vec model saved to {model_path}")
    
    return model

@st.cache_resource
def load_sbert_model():
    """Load SBERT model"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    
    # Check for available devices in order of preference: CUDA, MPS, CPU
    if torch.cuda.is_available():
        device = "cuda"
        st.info("Using CUDA GPU for SBERT")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        st.info("Using Apple MPS (Metal) for SBERT")
    else:
        device = "cpu"
        st.info("Using CPU for SBERT")
    
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model = model.to(device)
    return model

def get_doc_embedding_w2v(tokens, model):
    """Get document embedding using Word2Vec"""
    if not GENSIM_AVAILABLE or model is None:
        # Get vector size from model if available, otherwise default to 300
        vector_size = getattr(model, 'vector_size', 300) if model else 300
        return np.zeros(vector_size, dtype="float32")
    
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        # Use model's vector_size, default to 300 for Word2Vec
        vector_size = getattr(model, 'vector_size', 300)
        return np.zeros(vector_size, dtype="float32")
    return np.mean(vectors, axis=0)

@st.cache_data
def compute_job_embeddings_w2v(job_texts, _model):
    """Compute embeddings for jobs using Word2Vec"""
    embeddings = []
    for text in job_texts:
        tokens = simple_tokenize(text)
        emb = get_doc_embedding_w2v(tokens, _model)
        embeddings.append(emb)
    return np.array(embeddings)

@st.cache_data
def compute_job_embeddings_sbert(job_texts, _model, batch_size=None):
    """Compute embeddings for jobs using SBERT with optimized batch processing"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE or _model is None:
        return None
    
    num_texts = len(job_texts)
    if num_texts == 0:
        return np.array([])
    
    # Determine optimal batch size based on device
    if batch_size is None:
        try:
            import torch
            # Try to get device from model (SentenceTransformer models have _target_device or _modules)
            device = None
            if hasattr(_model, '_target_device'):
                device = _model._target_device
            elif hasattr(_model, '_modules') and len(_model._modules) > 0:
                # Get device from first module
                first_module = next(iter(_model._modules.values()))
                if hasattr(first_module, 'parameters'):
                    device = next(first_module.parameters()).device
            elif hasattr(_model, 'parameters'):
                device = next(_model.parameters()).device
            
            if device is not None and device.type == 'cuda':
                batch_size = 128  # Larger batch for GPU
            elif device is not None and device.type == 'mps':
                batch_size = 64   # Medium batch for MPS (Apple Silicon)
            else:
                batch_size = 32   # Smaller batch for CPU
        except:
            batch_size = 32
    
    batch_size = max(1, int(batch_size))
    
    # For large datasets, process in chunks
    if num_texts > 1000:
        # Process in chunks for better memory management
        chunk_size = max(batch_size * 10, 1000)  # Process 1000 texts at a time
        all_embeddings = []
        num_chunks = (num_texts + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_texts)
            chunk_texts = job_texts[start_idx:end_idx]
            
            # Compute embeddings for this chunk
            chunk_embeddings = _model.encode(
                chunk_texts,
                batch_size=batch_size,
                show_progress_bar=False,  # Disable internal progress bar for chunks
                convert_to_tensor=False,
            )
            all_embeddings.append(chunk_embeddings)
        
        return np.vstack(all_embeddings)
    else:
        # For smaller datasets, process all at once
        embeddings = _model.encode(
            job_texts,
            batch_size=batch_size,
            show_progress_bar=True,  # Show internal progress bar for smaller datasets
            convert_to_tensor=False,
        )
        
        return np.array(embeddings)

def save_job_embeddings(embeddings, method, num_jobs):
    """Save computed job embeddings to disk"""
    workspace_path = st.session_state.get('workspace_path')
    if workspace_path:
        models_dir = os.path.join(workspace_path, "models")
        os.makedirs(models_dir, exist_ok=True)
    else:
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
    
    # Create filename based on method and number of jobs
    method_short = "w2v" if method == "Word2Vec" else "sbert"
    embeddings_filename = f"job_embeddings_{method_short}_{num_jobs}_jobs.npy"
    metadata_filename = f"job_embeddings_{method_short}_{num_jobs}_jobs_metadata.joblib"
    
    embeddings_path = os.path.join(models_dir, embeddings_filename)
    metadata_path = os.path.join(models_dir, metadata_filename)
    
    try:
        # Save embeddings as numpy array
        np.save(embeddings_path, embeddings)
        
        # Save metadata
        if JOBLIB_AVAILABLE:
            metadata = {
                'method': method,
                'num_jobs': num_jobs,
                'embedding_shape': embeddings.shape,
                'saved_at': datetime.now().isoformat(),
                'embeddings_file': embeddings_filename
            }
            joblib.dump(metadata, metadata_path)
        
        return embeddings_path, metadata_path
    except Exception as e:
        st.error(f"Failed to save embeddings: {e}")
        return None, None

def load_job_embeddings(method, num_jobs):
    """Load saved job embeddings from disk"""
    workspace_path = st.session_state.get('workspace_path')
    if workspace_path:
        models_dir = os.path.join(workspace_path, "models")
    else:
        models_dir = "models"
    
    if not os.path.exists(models_dir):
        return None, None
    
    method_short = "w2v" if method == "Word2Vec" else "sbert"
    embeddings_filename = f"job_embeddings_{method_short}_{num_jobs}_jobs.npy"
    metadata_filename = f"job_embeddings_{method_short}_{num_jobs}jobs_metadata.joblib"
    
    embeddings_path = os.path.join(models_dir, embeddings_filename)
    metadata_path = os.path.join(models_dir, metadata_filename)
    
    if not os.path.exists(embeddings_path):
        return None, None
    
    try:
        embeddings = np.load(embeddings_path)
        metadata = None
        if JOBLIB_AVAILABLE and os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
        return embeddings, metadata
    except Exception as e:
        st.warning(f"Could not load saved embeddings: {e}")
        return None, None

def delete_job_embeddings(method, num_jobs):
    """Delete saved job embeddings from disk"""
    workspace_path = st.session_state.get('workspace_path')
    if workspace_path:
        models_dir = os.path.join(workspace_path, "models")
    else:
        models_dir = "models"
    
    if not os.path.exists(models_dir):
        return False
    
    method_short = "w2v" if method == "Word2Vec" else "sbert"
    embeddings_filename = f"job_embeddings_{method_short}_{num_jobs}_jobs.npy"
    metadata_filename = f"job_embeddings_{method_short}_{num_jobs}_jobs_metadata.joblib"
    
    embeddings_path = os.path.join(models_dir, embeddings_filename)
    metadata_path = os.path.join(models_dir, metadata_filename)
    
    deleted = False
    try:
        if os.path.exists(embeddings_path):
            os.remove(embeddings_path)
            deleted = True
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        return deleted
    except Exception as e:
        st.error(f"Failed to delete embeddings: {e}")
        return False

def list_all_saved_embeddings():
    """List all saved job embeddings files"""
    workspace_path = st.session_state.get('workspace_path')
    if workspace_path:
        models_dir = os.path.join(workspace_path, "models")
    else:
        models_dir = "models"
    
    if not os.path.exists(models_dir):
        return []
    
    embedding_files = []
    for filename in os.listdir(models_dir):
        if filename.startswith('job_embeddings_') and filename.endswith('.npy'):
            # Extract method and num_jobs from filename
            # Format: job_embeddings_{method}_{num_jobs}_jobs.npy
            parts = filename.replace('job_embeddings_', '').replace('.npy', '').split('_')
            if len(parts) >= 2:
                method_short = parts[0]
                num_jobs_str = parts[1].replace('jobs', '')
                try:
                    num_jobs = int(num_jobs_str)
                    method = "Word2Vec" if method_short == "w2v" else "Sentence-BERT (SBERT)"
                    
                    # Get metadata if available
                    metadata_filename = filename.replace('.npy', '_metadata.joblib')
                    metadata_path = os.path.join(models_dir, metadata_filename)
                    saved_at = "Unknown"
                    if os.path.exists(metadata_path) and JOBLIB_AVAILABLE:
                        try:
                            metadata = joblib.load(metadata_path)
                            saved_at = metadata.get('saved_at', 'Unknown')
                        except:
                            pass
                    
                    embedding_files.append({
                        'filename': filename,
                        'method': method,
                        'num_jobs': num_jobs,
                        'saved_at': saved_at,
                        'full_path': os.path.join(models_dir, filename)
                    })
                except ValueError:
                    continue
    
    return sorted(embedding_files, key=lambda x: x['saved_at'], reverse=True)

def normalize_job_field(value, default='N/A'):
    """Normalize job field values, converting NaN to default"""
    if pd.isna(value) or value == '' or str(value).lower() == 'nan':
        return default
    return value

@st.cache_data
def find_similar_jobs_w2v(query_text, job_texts, job_embeddings, df, valid_indices, top_k=5):
    """Find similar jobs using Word2Vec"""
    # Get the expected embedding dimension from job_embeddings
    if job_embeddings is not None and len(job_embeddings) > 0:
        if hasattr(job_embeddings, 'shape') and len(job_embeddings.shape) > 1:
            expected_dim = job_embeddings.shape[1]
        elif hasattr(job_embeddings, '__len__') and len(job_embeddings[0]) > 0:
            expected_dim = len(job_embeddings[0])
        else:
            expected_dim = 300  # Default Word2Vec dimension
    else:
        expected_dim = 300  # Default Word2Vec dimension
    
    query_tokens = simple_tokenize(query_text)
    # Use lowercase tokens for stable matching
    query_tokens = [t.lower() for t in query_tokens if t]
    query_token_set = set(query_tokens)

    query_emb = get_doc_embedding_w2v(query_tokens, st.session_state.get('w2v_model'))
    
    # Ensure query embedding matches job embeddings dimension
    if len(query_emb) != expected_dim:
        # If dimension mismatch, pad or truncate to match
        if len(query_emb) < expected_dim:
            # Pad with zeros
            query_emb = np.pad(query_emb, (0, expected_dim - len(query_emb)), mode='constant')
        else:
            # Truncate
            query_emb = query_emb[:expected_dim]
    
    query_emb = query_emb.reshape(1, -1)
    
    similarities = cosine_similarity(query_emb, job_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        # Use the original dataframe index from valid_indices
        original_idx = valid_indices[idx]
        job_row = df.loc[original_idx]

        # Simple keyword overlap between query and job title/skills to improve relevance
        title_text = str(job_row.get('Job Title') or '')
        skills_text = str(job_row.get('skills') or '')
        combined_text = f"{title_text} {skills_text}".lower()
        job_tokens = set(combined_text.split())

        if query_token_set:
            token_overlap = query_token_set & job_tokens
            keyword_score = len(token_overlap) / len(query_token_set)
        else:
            keyword_score = 0.0

        # Combine Word2Vec similarity with keyword overlap to stabilize rankings
        w2v_sim = float(similarities[idx])
        combined_score = 0.7 * w2v_sim + 0.3 * keyword_score

        results.append({
            'job_id': normalize_job_field(job_row.get('Job Id'), 'N/A'),
            'job_title': normalize_job_field(job_row.get('Job Title'), 'N/A'),
            'company': normalize_job_field(job_row.get('Company'), 'N/A'),
            'job_link': normalize_job_field(job_row.get('Job Link'), 'N/A'),
            'company_link': normalize_job_field(job_row.get('Company Link'), 'N/A'),
            'location': normalize_job_field(job_row.get('location'), 'N/A'),
            'country': normalize_job_field(job_row.get('Country'), 'N/A'),
            'salary_range': normalize_job_field(job_row.get('Salary Range'), 'N/A'),
            'experience': normalize_job_field(job_row.get('Experience'), 'N/A'),
            'benefits': normalize_job_field(job_row.get('Benefits'), 'N/A'),
            'skills': normalize_job_field(job_row.get('skills'), 'N/A'),
            'responsibilities': normalize_job_field(job_row.get('Responsibilities'), 'N/A'),
            'job_description': normalize_job_field(
                job_row.get('Job Description') or job_row.get('Description'),
                'N/A'
            ),
            'similarity': w2v_sim,
            'combined_score': combined_score
        })
    return results

@st.cache_data
def find_similar_jobs_sbert(query_text, job_texts, job_embeddings, _model, df, valid_indices, top_k=5):
    """Find similar jobs using SBERT"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE or _model is None:
        return []
    
    query_emb = _model.encode([query_text], batch_size=1, show_progress_bar=False, convert_to_tensor=False)[0]
    query_emb = query_emb.reshape(1, -1)
    
    similarities = cosine_similarity(query_emb, job_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        # Use the original dataframe index from valid_indices
        original_idx = valid_indices[idx]
        job_row = df.loc[original_idx]
        results.append({
            'job_id': normalize_job_field(job_row.get('Job Id'), 'N/A'),
            'job_title': normalize_job_field(job_row.get('Job Title'), 'N/A'),
            'company': normalize_job_field(job_row.get('Company'), 'N/A'),
            'job_link': normalize_job_field(job_row.get('Job Link'), 'N/A'),
            'company_link': normalize_job_field(job_row.get('Company Link'), 'N/A'),
            'location': normalize_job_field(job_row.get('location'), 'N/A'),
            'country': normalize_job_field(job_row.get('Country'), 'N/A'),
            'salary_range': normalize_job_field(job_row.get('Salary Range'), 'N/A'),
            'experience': normalize_job_field(job_row.get('Experience'), 'N/A'),
            'benefits': normalize_job_field(job_row.get('Benefits'), 'N/A'),
            'skills': normalize_job_field(job_row.get('skills'), 'N/A'),
            'responsibilities': normalize_job_field(job_row.get('Responsibilities'), 'N/A'),
            'job_description': normalize_job_field(
                job_row.get('Job Description') or job_row.get('Description'), 
                'N/A'
            ),
            'similarity': similarities[idx]
        })
    return results

# Functions to save/load NER results (must be defined before session state initialization)
def get_ner_results_path():
    """Get the path to save/load NER results"""
    workspace_path = st.session_state.get('workspace_path')
    if workspace_path:
        # Create models directory if it doesn't exist
        models_dir = os.path.join(workspace_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        return os.path.join(models_dir, "ner_results.json")
    else:
        # Fallback to current directory
        os.makedirs("models", exist_ok=True)
        return os.path.join("models", "ner_results.json")

def save_ner_results(results):
    """
    Save NER results to a JSON file.
    
    Args:
        results: Dictionary containing NER analysis results
    
    Returns:
        Path to saved file or None if save failed
    """
    try:
        results_path = get_ner_results_path()
        
        # Prepare results for JSON serialization
        # Convert Counter objects to lists of tuples
        save_data = {
            'total_entities': results.get('total_entities', 0),
            'unique_skills': results.get('unique_skills', 0),
            'unique_orgs': results.get('unique_orgs', 0),
            'skill_counts': results.get('skill_counts', []),
            'entity_counts': results.get('entity_counts', []),
            'skill_categories': results.get('skill_categories', []),
            'model_used': results.get('model_used', 'en_core_web_sm'),
            'config': results.get('config', {}),
            'saved_at': datetime.now().isoformat()
        }
        
        # Save to JSON file
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        return results_path
    except Exception as e:
        st.error(f"Error saving NER results: {e}")
        return None

def load_ner_results():
    """
    Load NER results from a JSON file.
    
    Returns:
        Dictionary containing NER results or None if file doesn't exist or load failed
    """
    try:
        results_path = get_ner_results_path()
        
        if not os.path.exists(results_path):
            return None
        
        # Load from JSON file
        with open(results_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        # Convert lists back to proper format
        results = {
            'total_entities': saved_data.get('total_entities', 0),
            'unique_skills': saved_data.get('unique_skills', 0),
            'unique_orgs': saved_data.get('unique_orgs', 0),
            'skill_counts': saved_data.get('skill_counts', []),
            'entity_counts': saved_data.get('entity_counts', []),
            'skill_categories': saved_data.get('skill_categories', []),
            'model_used': saved_data.get('model_used', 'en_core_web_sm'),
            'config': saved_data.get('config', {}),
            'saved_at': saved_data.get('saved_at', 'Unknown')
        }
        
        return results
    except Exception as e:
        # Silently fail on load - it's okay if file doesn't exist or is corrupted
        return None

# Page configuration
st.set_page_config(
    page_title="NLP Analytics - Job Analysis",
    page_icon="🤖",
    layout="wide"
)

# Load global CSS
try:
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles", "app.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

st.title("NLP Analytics")

# Professional introduction
st.markdown("""
<div style="margin-bottom: 2rem;">
    <p style="color: #6b7280; font-size: 1.1rem; line-height: 1.6;">
        Leverage advanced natural language processing to extract insights from job descriptions.
        Identify skills, topics, and patterns using state-of-the-art machine learning techniques.
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'jobs_df' not in st.session_state:
    st.session_state.jobs_df = None
if 'resumes_df' not in st.session_state:
    st.session_state.resumes_df = None
if 'cleaned_jobs_df' not in st.session_state:
    st.session_state.cleaned_jobs_df = None
if 'job_data_auto_load_attempted' not in st.session_state:
    st.session_state.job_data_auto_load_attempted = False
if 'ner_results' not in st.session_state:
    # Try to load saved NER results on page initialization
    saved_results = load_ner_results()
    st.session_state.ner_results = saved_results
    if saved_results:
        st.session_state.ner_results_loaded = True
    else:
        st.session_state.ner_results_loaded = False
if 'topic_model_results' not in st.session_state:
    st.session_state.topic_model_results = None
if 'embedding_results' not in st.session_state:
    st.session_state.embedding_results = None
if 'w2v_model' not in st.session_state:
    st.session_state.w2v_model = None
if 'sbert_model' not in st.session_state:
    st.session_state.sbert_model = None
if 'job_embeddings_w2v' not in st.session_state:
    st.session_state.job_embeddings_w2v = None
if 'job_embeddings_sbert' not in st.session_state:
    st.session_state.job_embeddings_sbert = None

# Function to load job data
@st.cache_data
def load_job_data():
    """Load job data from workspace"""
    workspace_path = st.session_state.get('workspace_path')
    if workspace_path:
        # Try loading from cleaned_data.json first (default for NLP)
        json_path = os.path.join(workspace_path, "Data", "cleaned_data.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            return df
        
        # Fallback to combined_data.json (has job links)
        json_path = os.path.join(workspace_path, "Data", "combined_data.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            return df
        else:
            # Fallback to cleaned job data (as used in notebooks)
            data_path = os.path.join(workspace_path, "Data_Cleaning", "cleaned_job_data_dedup.csv")
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                return df
            else:
                # Fallback to other locations
                data_path = os.path.join(workspace_path, "Data", "Jobs_data.csv")
                if os.path.exists(data_path):
                    df = pd.read_csv(data_path)
                    return df
    return None

# Auto-load cleaned job data if not already loaded
if st.session_state.cleaned_jobs_df is None and not st.session_state.job_data_auto_load_attempted:
    with st.spinner("🔄 Auto-loading cleaned job data..."):
        df = load_job_data()
        if df is not None:
            st.session_state.cleaned_jobs_df = df
            st.session_state.job_data_auto_load_attempted = True
            st.rerun()

# Function to load resume data
@st.cache_data
def load_resume_data():
    """Load resume data from workspace"""
    workspace_path = st.session_state.get('workspace_path')
    if workspace_path:
        resume_path = os.path.join(workspace_path, "Data_Cleaning", "cleaned_resume.csv")
        if os.path.exists(resume_path):
            df = pd.read_csv(resume_path)
            return df
    return None


# Tabs for different NLP tasks
tab1, tab2, tab3 = st.tabs([
    "Named Entity Recognition",
    "Topic Modeling",
    "Word Embeddings",
])

with tab1:
    st.markdown("### Named Entity Recognition (NER)")
    
    # Quick Start Guide
    with st.expander("📖 Quick Start Guide", expanded=True):
        st.markdown("""
        **Quick Start:**
        1. Click "Load Job Data" to load your dataset
        2. Select your preferred NER model (sm = fast, md = balanced, lg = best accuracy)
        3. Choose the number of job descriptions to analyze
        4. Click "Run NER Analysis" to extract skills and entities
        5. View results with skill categories and visualizations
        
        **What Gets Extracted:**
        - **Skills & Technologies**: 530+ programming languages, frameworks, and tools
        - **Companies & Organizations**: Company names and related entities
        - **Locations**: Cities, countries, and work settings
        - **Other Entities**: Dates, persons, and more
        """)
    
    # About the Models
    with st.expander("ℹ️ About the Models"):
        st.markdown("""
        **en_core_web_sm (Small Model)**
        - **Speed**: Fast processing (< 1 second per 100 texts)
        - **Accuracy**: Good for most use cases
        - **Size**: ~50MB
        - **Use case**: Large datasets, quick analysis, production environments
        
        **en_core_web_md (Medium Model)**
        - **Speed**: Moderate processing (~2-3 seconds per 100 texts)
        - **Accuracy**: Better entity recognition with word vectors
        - **Size**: ~40MB + vectors
        - **Use case**: Balanced accuracy and speed
        
        **en_core_web_lg (Large Model)**
        - **Speed**: Slower processing (~5-10 seconds per 100 texts)
        - **Accuracy**: Best accuracy with comprehensive word vectors
        - **Size**: ~560MB + vectors
        - **Use case**: High-accuracy requirements, research applications
        """)
    
    if st.session_state.jobs_df is None:
        if st.button("Load Job Data", key="ner_load"):
            with st.spinner("Loading job data..."):
                df = load_job_data()
                if df is not None:
                    st.session_state.jobs_df = df
                    st.success(f"✅ Loaded {len(df):,} job postings")
                    st.rerun()
                else:
                    st.error("❌ Could not load data")
    else:
        df = st.session_state.jobs_df
        
        # NER Configuration
        st.markdown("---")
        st.markdown("#### ⚙️ Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Model selection with mapping
            model_option = st.selectbox(
                "Select NER Model",
                ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
                index=0,
                help="""Choose spaCy model:
                - **sm**: Small (fast, ~50MB) - Recommended for large datasets
                - **md**: Medium (balanced, ~40MB + vectors)
                - **lg**: Large (best accuracy, ~560MB + vectors) - Slower for large datasets"""
            )
            model_name = model_option
        
        with col2:
            sample_size = st.number_input(
                "Sample Size",
                min_value=10,
                max_value=len(df),
                value=min(100, len(df)),
                help="Number of job descriptions to analyze"
            )
            
        with col3:
            use_batch = st.checkbox(
                "Use Batch Processing",
                value=True,
                help="Process texts in batches for better performance (recommended for large datasets)"
            )
        
        # Batch size configuration for large datasets
        if sample_size > 1000:
            batch_size = st.slider(
                "Batch Size",
                min_value=100,
                max_value=2000,
                value=min(1000, max(500, sample_size // 15)),  # Adaptive default
                step=100,
                help="Number of texts to process at once. Larger batches = faster but more memory. Recommended: 500-1000"
            )
        else:
            batch_size = 500  # Default for smaller datasets
        
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2, col3 = st.columns(3)
            with col1:
                use_word_boundaries = st.checkbox(
                    "Word Boundary Check",
                    value=True,
                    help="Prevent partial matches (e.g., 'java' in 'javascript')"
                )
            with col2:
                use_overlap_resolution = st.checkbox(
                    "Overlap Resolution",
                    value=True,
                    help="Prefer longer matches over shorter ones"
                )
            with col3:
                use_context_filter = st.checkbox(
                    "Context Filtering",
                    value=False,
                    help="Only extract skills in relevant contexts (may reduce recall)"
                )

            # Parallel workers for spaCy (n_process)
            st.markdown("##### Performance")
            n_process = st.slider(
                "Parallel workers (CPU cores)",
                min_value=1,
                max_value=20,
                value=1,
                help="Use multiple CPU processes for spaCy NER. 1 = no parallelism. Higher values can speed up large batches but use more memory."
            )
        
        if st.button("Run NER Analysis", type="primary"):
            if not SPACY_AVAILABLE:
                st.error("spaCy is not available. Please install requirements-nlp.txt")
            else:
                with st.spinner("Running Enhanced Named Entity Recognition..."):
                    # Load models with selected model
                    nlp = load_spacy_model(model_name)
                    if nlp is None:
                        st.error(f"Could not load spaCy model '{model_name}'. Trying fallback...")
                        nlp = load_spacy_model("en_core_web_sm")
                    
                    skill_matcher = build_skill_ner(MASTER_SKILL_LIST, _nlp=nlp)
                    
                    if nlp is None or skill_matcher is None:
                        st.error("Could not initialize NER models")
                    else:
                        # Sample some job descriptions
                        text_column = 'job_text_cleaned' if 'job_text_cleaned' in df.columns else 'Job Description'
                        if text_column not in df.columns:
                            st.error(f"Could not find job text column. Available columns: {list(df.columns)}")
                            st.stop()
                        
                        sample_texts = df[text_column].dropna().sample(min(sample_size, len(df))).tolist()
                        
                        all_skills = []
                        all_spacy_entities = []
                        skill_categories = Counter()
                        
                        # Use configured batch size
                        optimal_batch_size = batch_size
                        
                        # Process texts with progress indication
                        if use_batch and len(sample_texts) > 10:
                            # Use optimized batch processing
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            def update_progress(processed, total):
                                progress = processed / total
                                progress_bar.progress(progress)
                                status_text.text(f"Processing: {processed:,} / {total:,} texts ({progress*100:.1f}%)")
                            
                            # Extract skills in batch
                            status_text.text("Extracting skills from job descriptions...")
                            skill_lists = extract_skills_batch(
                                sample_texts, skill_matcher, nlp=nlp,
                                batch_size=optimal_batch_size,
                                use_word_boundaries=use_word_boundaries,
                                use_overlap_resolution=use_overlap_resolution,
                                use_context_filter=use_context_filter,
                                progress_callback=update_progress,
                                n_process=n_process,
                            )
                            all_skills = [skill for skills in skill_lists for skill in skills]
                            
                            # Extract spaCy entities in batch
                            status_text.text("Extracting spaCy entities (ORG, GPE, etc.)...")
                            progress_bar.progress(0)
                            entity_lists = extract_spacy_entities_batch(
                                sample_texts, nlp=nlp,
                                batch_size=optimal_batch_size,
                                progress_callback=update_progress,
                                n_process=n_process,
                            )
                            all_spacy_entities = [ent for ents in entity_lists for ent in ents]
                            
                            progress_bar.empty()
                            status_text.empty()
                        else:
                            # Process individually (for very small datasets)
                            progress_bar = st.progress(0)
                            for idx, text in enumerate(sample_texts):
                                skills = extract_skill_entities(
                                    text, skill_matcher, nlp=nlp,
                                    use_word_boundaries=use_word_boundaries,
                                    use_overlap_resolution=use_overlap_resolution,
                                    use_context_filter=use_context_filter
                                )
                                all_skills.extend(skills)
                                
                                spacy_ents = extract_spacy_entities(text, nlp=nlp)
                                all_spacy_entities.extend(spacy_ents)
                                
                                progress_bar.progress((idx + 1) / len(sample_texts))
                            progress_bar.empty()
                        
                        # Count frequencies
                        skill_counts = Counter(all_skills)
                        entity_counts = Counter([label for _, label in all_spacy_entities])
                        
                        # Categorize skills
                        for skill, count in skill_counts.items():
                            category = categorize_skill(skill)
                            skill_categories[category] += count
                        
                        # Store results
                        st.session_state.ner_results = {
                            'total_entities': len(all_spacy_entities),
                            'unique_skills': len(skill_counts),
                            'unique_orgs': entity_counts.get('ORG', 0),
                            'skill_counts': skill_counts.most_common(30),
                            'entity_counts': entity_counts.most_common(15),
                            'skill_categories': skill_categories.most_common(),
                            'model_used': model_name,
                            'config': {
                                'word_boundaries': use_word_boundaries,
                                'overlap_resolution': use_overlap_resolution,
                                'context_filter': use_context_filter,
                                'batch_processing': use_batch
                            }
                        }
                        
                        # Save results to file
                        results_path = save_ner_results(st.session_state.ner_results)
                        if results_path:
                            st.session_state.ner_results_loaded = False  # Mark as newly generated
                            # Show relative path for better readability
                            workspace_path = st.session_state.get('workspace_path')
                            if workspace_path and results_path.startswith(workspace_path):
                                rel_path = os.path.relpath(results_path, workspace_path)
                                st.success(f"✅ Enhanced NER Analysis completed! Results saved to `{rel_path}`")
                            else:
                                st.success(f"✅ Enhanced NER Analysis completed! Results saved to `{results_path}`")
                        else:
                            st.success("✅ Enhanced NER Analysis completed!")
        
        if st.session_state.ner_results:
            results = st.session_state.ner_results
            
            # Show if results were loaded from file or newly generated
            if st.session_state.get('ner_results_loaded', False):
                saved_at = results.get('saved_at', 'Unknown')
                st.info(f"**Loaded saved results** (saved at: {saved_at})")
            else:
                st.info("**Fresh analysis results**")
            
            # Add option to reload saved results
            col_reload, col_clear = st.columns([1, 1])
            with col_reload:
                if st.button("Reload Saved Results", help="Reload the last saved NER results from file"):
                    saved_results = load_ner_results()
                    if saved_results:
                        st.session_state.ner_results = saved_results
                        st.session_state.ner_results_loaded = True
                        st.success("Reloaded saved results!")
                        st.rerun()
                    else:
                        st.error("❌ No saved results found")
            
            with col_clear:
                if st.button("Clear Results", help="Clear current results from memory"):
                    st.session_state.ner_results = None
                    st.session_state.ner_results_loaded = False
                    st.success("Results cleared!")
                    st.rerun()
            
            st.markdown("#### NER Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Entities", f"{results['total_entities']:,}")
            with col2:
                st.metric("Unique Skills", f"{results['unique_skills']:,}")
            with col3:
                st.metric("Unique Organizations", f"{results['unique_orgs']:,}")
            with col4:
                model_display = results.get('model_used', 'en_core_web_sm')
                st.metric("Model Used", model_display.replace('en_core_web_', '').upper())
            
            # Configuration used
            if 'config' in results:
                config = results['config']
                st.info(f"**Configuration**: Word Boundaries: {config['word_boundaries']} | "
                       f"Overlap Resolution: {config['overlap_resolution']} | "
                       f"Context Filter: {config['context_filter']} | "
                       f"Batch Processing: {config['batch_processing']}")
            
            # Skill categories visualization
            if 'skill_categories' in results and results['skill_categories']:
                st.markdown("**Skills by Category:**")
                cat_df = pd.DataFrame(results['skill_categories'], columns=['Category', 'Count'])
                fig_cat = px.bar(
                    cat_df, 
                    x='Category', 
                    y='Count', 
                    title='Skill Distribution by Category',
                    color='Count',
                    color_continuous_scale='Blues'
                )
                fig_cat.update_xaxes(tickangle=45)
                st.plotly_chart(fig_cat, use_container_width=True)
            
            # Top skills
            if 'skill_counts' in results:
                st.markdown("**Top Skills:**")
                skill_df = pd.DataFrame(results['skill_counts'], columns=['Skill', 'Count'])
                
                # Add category column
                skill_df['Category'] = skill_df['Skill'].apply(categorize_skill)
                
                # Display with tabs for different views
                tab_skills, tab_categories = st.tabs(["All Skills", "By Category"])
                
                with tab_skills:
                    st.dataframe(skill_df[['Skill', 'Count', 'Category']], use_container_width=True, height=400)
                
                with tab_categories:
                    # Group by category
                    for category in skill_df['Category'].unique():
                        cat_skills = skill_df[skill_df['Category'] == category].head(10)
                        if len(cat_skills) > 0:
                            st.markdown(f"**{category}** ({len(cat_skills)} skills)")
                            st.dataframe(cat_skills[['Skill', 'Count']], use_container_width=True)
            
            # Entity distribution
            if 'entity_counts' in results:
                st.markdown("**Entity Types Distribution:**")
                ent_df = pd.DataFrame(results['entity_counts'], columns=['Entity Type', 'Count'])
                fig_ent = px.bar(
                    ent_df, 
                    x='Entity Type', 
                    y='Count', 
                    title='SpaCy Entity Type Distribution',
                    color='Count',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_ent, use_container_width=True)
            
            # Top skills word cloud style visualization
            if 'skill_counts' in results and len(results['skill_counts']) > 0:
                st.markdown("**Top Skills Visualization:**")
                
                # Limit to top 20 but keep dynamic sizing so all bars stay visible
                top_skills = results['skill_counts'][:20]
                top_skills_df = pd.DataFrame(top_skills, columns=['Skill', 'Count'])
                bar_height = max(420, len(top_skills_df) * 28)
                fig_skills = px.bar(
                    top_skills_df,
                    x='Count',
                    y='Skill',
                    orientation='h',
                    title='Top 20 Skills',
                    color='Count',
                    color_continuous_scale='Greens',
                    height=bar_height
                )
                fig_skills.update_layout(
                    yaxis={'categoryorder': 'total ascending', 'automargin': True},
                    margin=dict(l=160, r=40, t=60, b=40)
                )
                st.plotly_chart(fig_skills, use_container_width=True)

with tab2:
    st.markdown("### Topic Modeling")
    st.markdown("""
    Discover hidden themes and topics in job descriptions using:
    - **Latent Dirichlet Allocation (LDA)**: Probabilistic topic modeling
    - **Latent Semantic Analysis (LSA)**: SVD-based topic extraction
    """)
    
    if st.session_state.cleaned_jobs_df is None:
        if st.button("Load Job Data", key="topic_load"):
            with st.spinner("Loading job data..."):
                df = load_job_data()
                if df is not None:
                    st.session_state.cleaned_jobs_df = df
                    st.success(f"✅ Loaded {len(df):,} job postings")
                    st.rerun()
                else:
                    st.error("❌ Could not load data")
    else:
        df = st.session_state.cleaned_jobs_df
        st.success(f"✅ Working with {len(df):,} job postings")
        
        # Topic Modeling Settings
        col1, col2, col3 = st.columns(3)
        with col1:
            method = st.selectbox(
                "Topic Modeling Method",
                ["LDA", "LSA"],
                help="Choose the topic modeling algorithm"
            )
        with col2:
            num_topics = st.slider(
                "Number of Topics",
                min_value=3,
                max_value=100,
                value=10,
                help="Number of topics to extract"
            )
        with col3:
            num_words = st.slider(
                "Words per Topic",
                min_value=5,
                max_value=20,
                value=10,
                help="Number of top words to display per topic"
            )
        
        # Parallel CPU configuration
        available_cores = max(1, os.cpu_count() or 1)
        cpu_cores = st.slider(
            "CPU cores for parallel LDA",
            min_value=1,
            max_value=available_cores,
            value=min(available_cores, 4),
            help="Number of CPU processes scikit-learn should use for LDA (set to 1 to disable parallelism)"
        )
        
        # Model saving option
        save_model = st.checkbox(
            "Save trained model for future use",
            value=True,
            help="Save the trained topic model to disk for faster loading on subsequent runs"
        )
        
        if st.button("Run Topic Modeling", type="primary"):
            if not SKLEARN_AVAILABLE:
                st.error("scikit-learn is not available. Please install requirements.")
            else:
                with st.spinner(f"Running {method} topic modeling..."):
                    # Get job texts
                    text_column = 'job_text_cleaned' if 'job_text_cleaned' in df.columns else 'Job Description'
                    if text_column not in df.columns:
                        st.error(f"Could not find job text column. Available columns: {list(df.columns)}")
                        st.stop()
                    texts = df[text_column].dropna().tolist()
                    
                    # Run topic modeling
                    results = run_topic_modeling(texts, method, num_topics, num_words, save_model, cpu_cores)
                    
                    if results:
                        st.session_state.topic_model_results = results
                        st.success(f"✅ {method} topic modeling completed!")
                        st.rerun()
                    else:
                        st.error("Topic modeling failed")
        
        if st.session_state.topic_model_results:
            results = st.session_state.topic_model_results
            st.markdown("#### Topic Modeling Results")
            st.success(f"✅ Extracted {results['n_topics']} topics using {results['method']}")
            
            if 'vocab_size' in results:
                st.info(f"Vocabulary size: {results['vocab_size']:,}")
            if 'explained_variance' in results:
                st.info(f"Explained variance: {results['explained_variance']:.3f}")
            
            # Show topics
            st.markdown("**Discovered Topics:**")
            for i, words in enumerate(results['topic_words'][:min(5, len(results['topic_words']))]):
                st.write(f"**Topic {i+1}:** {', '.join(words)}")
            
            if len(results['topic_words']) > 5:
                with st.expander("Show all topics"):
                    for i, words in enumerate(results['topic_words']):
                        st.write(f"**Topic {i+1}:** {', '.join(words)}")
        
        # Show saved models
        st.markdown("#### Saved Models")
        workspace_path = st.session_state.get('workspace_path')
        if workspace_path:
            models_dir = os.path.join(workspace_path, "models")
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.startswith('topic_model_') and f.endswith('.joblib')]
                if model_files:
                    st.markdown("**Available saved models:**")
                    for model_file in sorted(model_files):
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.write(f"📁 {model_file}")
                        with col2:
                            if st.button("Load", key=f"load_{model_file}"):
                                model_path = os.path.join(models_dir, model_file)
                                try:
                                    saved_data = joblib.load(model_path)
                                    st.session_state.topic_model_results = saved_data['results']
                                    st.success(f"✅ Loaded {model_file}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to load model: {e}")
                        with col3:
                            if st.button("Delete", key=f"delete_{model_file}"):
                                model_path = os.path.join(models_dir, model_file)
                                try:
                                    os.remove(model_path)
                                    st.success(f"🗑️ Deleted {model_file}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to delete: {e}")
                else:
                    st.info("No saved topic models found.")
            else:
                st.info("Models directory not found.")

with tab3:
    st.markdown("### Word Embeddings")
    st.markdown("""
    Analyze semantic relationships between words and find similar jobs using:
    - **Word2Vec**: Neural word embeddings
    - **Sentence-BERT (SBERT)**: Sentence-level embeddings for job matching
    """)
    
    if st.session_state.cleaned_jobs_df is None:
        if st.button("Load Job Data", key="embedding_load_jobs"):
            with st.spinner("Loading job data..."):
                df = load_job_data()
                if df is not None:
                    st.session_state.cleaned_jobs_df = df
                    st.success(f"✅ Loaded {len(df):,} job postings")
                    st.rerun()
                else:
                    st.error("❌ Could not load data")
    else:
        df = st.session_state.cleaned_jobs_df
        st.success(f"✅ Working with {len(df):,} job postings")
        
        # Load resume data for training
        if st.session_state.resumes_df is None:
            if st.button("Load Resume Data", key="embedding_load_resumes"):
                with st.spinner("Loading resume data..."):
                    resume_df = load_resume_data()
                    if resume_df is not None:
                        st.session_state.resumes_df = resume_df
                        st.success(f"✅ Loaded {len(resume_df):,} resumes")
                        st.rerun()
                    else:
                        st.error("❌ Could not load resume data")
        else:
            resume_df = st.session_state.resumes_df
            st.success(f"✅ Working with {len(resume_df):,} resumes")
        
        # Word Embedding Settings
        embedding_method = st.selectbox(
            "Embedding Method",
            ["Word2Vec", "Sentence-BERT (SBERT)"],
            help="Choose the embedding method"
        )
        
        # Parallel CPU configuration for embedding models
        available_cores = max(20, os.cpu_count() or 1)
        emb_cpu_cores = st.slider(
            "CPU cores for embeddings",
            min_value=1,
            max_value=available_cores,
            value=min(available_cores, 4),
            help="Number of CPU cores to use for Word2Vec training and CPU-based SBERT encoding"
        )
        
        # Model saving option for Word2Vec
        if embedding_method == "Word2Vec":
            save_w2v_model = st.checkbox(
                "Save trained Word2Vec model for future use",
                value=True,
                help="Save the trained Word2Vec model to disk for faster loading on subsequent runs"
            )
        
        # Load/Train models
        if embedding_method == "Word2Vec":
            if st.session_state.w2v_model is None:
                # If a saved model exists, offer to load it before training
                workspace_path = st.session_state.get('workspace_path')
                models_dir = None
                saved_w2v_path = None
                if workspace_path:
                    models_dir = os.path.join(workspace_path, "models")
                else:
                    models_dir = "models"
                if os.path.exists(models_dir):
                    candidate_path = os.path.join(models_dir, "word2vec_model.joblib")
                    if os.path.exists(candidate_path) and JOBLIB_AVAILABLE:
                        saved_w2v_path = candidate_path
                
                if saved_w2v_path:
                    st.info(f"📁 Found saved Word2Vec model at `{os.path.basename(saved_w2v_path)}`.")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Load Saved Word2Vec Model", key="load_w2v_before_train"):
                            try:
                                saved_data = joblib.load(saved_w2v_path)
                                st.session_state.w2v_model = saved_data['model']
                                st.success("✅ Loaded saved Word2Vec model!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to load saved model: {e}")
                    with col2:
                        if st.button("Train New Word2Vec Model", key="train_w2v"):
                            if not GENSIM_AVAILABLE:
                                st.error("Gensim not available")
                            elif st.session_state.resumes_df is None:
                                st.error("Please load resume data first")
                            else:
                                with st.spinner("Training Word2Vec model..."):
                                    text_column = 'job_text_cleaned' if 'job_text_cleaned' in df.columns else 'Job Description'
                                    job_texts = df[text_column].dropna().tolist()
                                    resume_texts = resume_df['cleaned_text'].dropna().tolist()
                                    model = train_word2vec_model(
                                        job_texts,
                                        resume_texts,
                                        save_model=save_w2v_model,
                                        workers=emb_cpu_cores,
                                    )
                                    if model:
                                        st.session_state.w2v_model = model
                                        st.success("✅ Word2Vec model trained!")
                                        st.rerun()
                else:
                    if st.button("Train Word2Vec Model", key="train_w2v"):
                        if not GENSIM_AVAILABLE:
                            st.error("Gensim not available")
                        elif st.session_state.resumes_df is None:
                            st.error("Please load resume data first")
                        else:
                            with st.spinner("Training Word2Vec model..."):
                                text_column = 'job_text_cleaned' if 'job_text_cleaned' in df.columns else 'Job Description'
                                job_texts = df[text_column].dropna().tolist()
                                resume_texts = resume_df['cleaned_text'].dropna().tolist()
                                model = train_word2vec_model(
                                    job_texts,
                                    resume_texts,
                                    save_model=save_w2v_model,
                                    workers=emb_cpu_cores,
                                )
                                if model:
                                    st.session_state.w2v_model = model
                                    st.success("✅ Word2Vec model trained!")
                                    st.rerun()
            else:
                st.success("✅ Word2Vec model ready")
                
                # Show saved models
                st.markdown("#### Saved Word2Vec Models")
                workspace_path = st.session_state.get('workspace_path')
                if workspace_path:
                    models_dir = os.path.join(workspace_path, "models")
                    if os.path.exists(models_dir):
                        w2v_files = [f for f in os.listdir(models_dir) if f == 'word2vec_model.joblib']
                        if w2v_files:
                            st.markdown("**Available saved Word2Vec model:**")
                            model_file = w2v_files[0]
                            col1, col2, col3 = st.columns([3, 1, 1])
                            with col1:
                                st.write(f"📁 {model_file}")
                            with col2:
                                if st.button("Load", key="load_w2v"):
                                    model_path = os.path.join(models_dir, model_file)
                                    try:
                                        saved_data = joblib.load(model_path)
                                        st.session_state.w2v_model = saved_data['model']
                                        st.success(f"✅ Loaded {model_file}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to load model: {e}")
                            with col3:
                                if st.button("Delete", key="delete_w2v"):
                                    model_path = os.path.join(models_dir, model_file)
                                    try:
                                        os.remove(model_path)
                                        st.success(f"🗑️ Deleted {model_file}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to delete: {e}")
                        else:
                            st.info("No saved Word2Vec model found.")
                    else:
                        st.info("Models directory not found.")
                
                # Compute embeddings
                if st.session_state.job_embeddings_w2v is None:
                    text_column = 'job_text_cleaned' if 'job_text_cleaned' in df.columns else 'Job Description'
                    job_texts = df[text_column].dropna().tolist()
                    num_jobs = len(job_texts)
                    
                    # Check for saved embeddings
                    saved_embeddings, saved_metadata = load_job_embeddings("Word2Vec", num_jobs)
                    if saved_embeddings is not None:
                        st.info(f"📁 Found saved embeddings ({num_jobs} jobs, saved at: {saved_metadata.get('saved_at', 'Unknown') if saved_metadata else 'Unknown'})")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("Load Saved Embeddings", key="load_w2v_emb"):
                                st.session_state.job_embeddings_w2v = saved_embeddings
                                st.success("✅ Loaded saved embeddings!")
                                st.rerun()
                        with col2:
                            if st.button("Recompute Embeddings", key="recompute_w2v_emb"):
                                with st.spinner("Computing job embeddings..."):
                                    embeddings = compute_job_embeddings_w2v(job_texts, st.session_state.w2v_model)
                                    st.session_state.job_embeddings_w2v = embeddings
                                    # Auto-save after computation
                                    save_job_embeddings(embeddings, "Word2Vec", num_jobs)
                                    st.success("✅ Job embeddings computed and saved!")
                                    st.rerun()
                        with col3:
                            if st.button("🗑️ Delete Saved Embeddings", key="delete_w2v_emb"):
                                if delete_job_embeddings("Word2Vec", num_jobs):
                                    st.success("🗑️ Deleted saved embeddings!")
                                    st.rerun()
                                else:
                                    st.error("Failed to delete embeddings")
                    else:
                        if st.button("Compute Job Embeddings", key="compute_w2v_emb"):
                            with st.spinner("Computing job embeddings..."):
                                embeddings = compute_job_embeddings_w2v(job_texts, st.session_state.w2v_model)
                                st.session_state.job_embeddings_w2v = embeddings
                                # Auto-save after computation
                                save_job_embeddings(embeddings, "Word2Vec", num_jobs)
                                st.success("✅ Job embeddings computed and saved!")
                                st.rerun()
                else:
                    st.success("✅ Job embeddings ready")
                    text_column = 'job_text_cleaned' if 'job_text_cleaned' in df.columns else 'Job Description'
                    job_texts = df[text_column].dropna().tolist()
                    num_jobs = len(job_texts)
                    
                    # Save embeddings option
                    if st.button("Save Embeddings to File", key="save_w2v_emb"):
                        embeddings_path, metadata_path = save_job_embeddings(
                            st.session_state.job_embeddings_w2v,
                            "Word2Vec",
                            num_jobs
                        )
                        if embeddings_path:
                            st.success(f"✅ Embeddings saved to {os.path.basename(embeddings_path)}")
                        else:
                            st.error("Failed to save embeddings")
        
        elif embedding_method == "Sentence-BERT (SBERT)":
            if st.session_state.sbert_model is None:
                # If a saved SBERT model exists locally, offer the same load/new pattern as Word2Vec
                workspace_path = st.session_state.get('workspace_path')
                models_dir = None
                saved_sbert_path = None
                if workspace_path:
                    models_dir = os.path.join(workspace_path, "models")
                else:
                    models_dir = "models"
                # We expect a locally fine‑tuned SBERT model to be saved as a directory
                # (using SentenceTransformer.save()), e.g. models/sbert_model
                candidate_path = os.path.join(models_dir, "sbert_model")
                if os.path.isdir(candidate_path) and SENTENCE_TRANSFORMERS_AVAILABLE:
                    saved_sbert_path = candidate_path

                if saved_sbert_path:
                    st.info(f"📁 Found saved SBERT model at `{os.path.basename(saved_sbert_path)}`.")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Load Saved SBERT Model", key="load_sbert_saved"):
                            try:
                                # Load SBERT model from local directory
                                from sentence_transformers import SentenceTransformer
                                model = SentenceTransformer(saved_sbert_path)
                                st.session_state.sbert_model = model
                                st.success("✅ Loaded saved SBERT model!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to load saved SBERT model: {e}")
                    with col2:
                        if st.button("Load Pretrained SBERT Model", key="load_sbert_pretrained"):
                            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                                st.error("Sentence Transformers not available")
                            else:
                                with st.spinner("Loading SBERT model from HuggingFace..."):
                                    model = load_sbert_model()
                                    if model:
                                        st.session_state.sbert_model = model
                                        st.success("✅ SBERT model loaded!")
                                        st.rerun()
                else:
                    if st.button("Load SBERT Model", key="load_sbert"):
                        if not SENTENCE_TRANSFORMERS_AVAILABLE:
                            st.error("Sentence Transformers not available")
                        else:
                            with st.spinner("Loading SBERT model..."):
                                model = load_sbert_model()
                                if model:
                                    st.session_state.sbert_model = model
                                    st.success("✅ SBERT model loaded!")
                                    st.rerun()
            else:
                st.success("✅ SBERT model ready")
                
                # Compute embeddings
                if st.session_state.job_embeddings_sbert is None:
                    text_column = 'job_text_cleaned' if 'job_text_cleaned' in df.columns else 'Job Description'
                    job_texts = df[text_column].dropna().tolist()
                    num_jobs = len(job_texts)
                    
                    # Check for saved embeddings
                    saved_embeddings, saved_metadata = load_job_embeddings("Sentence-BERT (SBERT)", num_jobs)
                    if saved_embeddings is not None:
                        st.info(f"📁 Found saved embeddings ({num_jobs} jobs, saved at: {saved_metadata.get('saved_at', 'Unknown') if saved_metadata else 'Unknown'})")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("Load Saved Embeddings", key="load_sbert_emb"):
                                st.session_state.job_embeddings_sbert = saved_embeddings
                                st.success("✅ Loaded saved embeddings!")
                                st.rerun()
                        with col2:
                            if st.button("Recompute Embeddings", key="recompute_sbert_emb"):
                                with st.spinner(f"Computing embeddings for {num_jobs} jobs..."):
                                    embeddings = compute_job_embeddings_sbert(
                                        job_texts,
                                        st.session_state.sbert_model,
                                        batch_size=None,  # Auto-detect optimal batch size
                                    )
                                if embeddings is not None:
                                    st.session_state.job_embeddings_sbert = embeddings
                                    # Auto-save after computation
                                    save_job_embeddings(embeddings, "Sentence-BERT (SBERT)", num_jobs)
                                    st.success("✅ Job embeddings computed and saved!")
                                    st.rerun()
                                else:
                                    st.error("Failed to compute embeddings")
                        with col3:
                            if st.button("🗑️ Delete Saved Embeddings", key="delete_sbert_emb"):
                                if delete_job_embeddings("Sentence-BERT (SBERT)", num_jobs):
                                    st.success("🗑️ Deleted saved embeddings!")
                                    st.rerun()
                                else:
                                    st.error("Failed to delete embeddings")
                    else:
                        if st.button("Compute Job Embeddings", key="compute_sbert_emb"):
                            with st.spinner(f"Computing embeddings for {num_jobs} jobs..."):
                                embeddings = compute_job_embeddings_sbert(
                                    job_texts,
                                    st.session_state.sbert_model,
                                    batch_size=None,  # Auto-detect optimal batch size
                                )
                            if embeddings is not None:
                                st.session_state.job_embeddings_sbert = embeddings
                                # Auto-save after computation
                                save_job_embeddings(embeddings, "Sentence-BERT (SBERT)", num_jobs)
                                st.success("✅ Job embeddings computed and saved!")
                                st.rerun()
                            else:
                                st.error("Failed to compute embeddings")
                else:
                    st.success("✅ Job embeddings ready")
                    text_column = 'job_text_cleaned' if 'job_text_cleaned' in df.columns else 'Job Description'
                    job_texts = df[text_column].dropna().tolist()
                    num_jobs = len(job_texts)
                    
                    # Save embeddings option
                    if st.button("Save Embeddings to File", key="save_sbert_emb"):
                        embeddings_path, metadata_path = save_job_embeddings(
                            st.session_state.job_embeddings_sbert,
                            "Sentence-BERT (SBERT)",
                            num_jobs
                        )
                        if embeddings_path:
                            st.success(f"✅ Embeddings saved to {os.path.basename(embeddings_path)}")
                        else:
                            st.error("Failed to save embeddings")
        
        # Manage Saved Embeddings Section
        st.markdown("---")
        st.markdown("#### Manage Saved Embeddings")
        with st.expander("View and Delete All Saved Embeddings", expanded=False):
            saved_embeddings_list = list_all_saved_embeddings()
            if saved_embeddings_list:
                st.write(f"**Found {len(saved_embeddings_list)} saved embedding file(s):**")
                for emb_info in saved_embeddings_list:
                    col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])
                    with col1:
                        st.write(f"📁 {emb_info['filename']}")
                    with col2:
                        st.write(f"**Method:** {emb_info['method']}")
                    with col3:
                        st.write(f"**Jobs:** {emb_info['num_jobs']:,} | **Saved:** {emb_info['saved_at'][:10] if emb_info['saved_at'] != 'Unknown' else 'Unknown'}")
                    with col4:
                        if st.button("Load", key=f"load_emb_{emb_info['filename']}"):
                            try:
                                saved_embeddings, saved_metadata = load_job_embeddings(emb_info['method'], emb_info['num_jobs'])
                                if saved_embeddings is not None:
                                    # Set the appropriate session state variable based on method
                                    if emb_info['method'] == "Word2Vec":
                                        st.session_state.job_embeddings_w2v = saved_embeddings
                                    elif emb_info['method'] == "Sentence-BERT (SBERT)":
                                        st.session_state.job_embeddings_sbert = saved_embeddings
                                    st.success(f"Loaded {emb_info['filename']}")
                                    st.rerun()
                                else:
                                    st.error("Failed to load embeddings")
                            except Exception as e:
                                st.error(f"Failed to load embeddings: {e}")
                    with col5:
                        if st.button("Delete", key=f"delete_emb_{emb_info['filename']}"):
                            if delete_job_embeddings(emb_info['method'], emb_info['num_jobs']):
                                st.success(f"Deleted {emb_info['filename']}")
                                st.rerun()
                            else:
                                st.error("Failed to delete")
                st.markdown("---")
                if st.button("Delete All Saved Embeddings", key="delete_all_embeddings", type="secondary"):
                    deleted_count = 0
                    for emb_info in saved_embeddings_list:
                        if delete_job_embeddings(emb_info['method'], emb_info['num_jobs']):
                            deleted_count += 1
                    if deleted_count > 0:
                        st.success(f"Deleted {deleted_count} embedding file(s)")
                        st.rerun()
            else:
                st.info("No saved embeddings found.")
    