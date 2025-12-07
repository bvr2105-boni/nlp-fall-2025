"""
Functions module for NLP Fall 2025 Project
==========================================

This module provides reusable functions for:
- Vector search and embeddings
- LLM-based evaluation
- Resume and job matching
- NLP processing and analysis

Main modules:
- vector_search_eval: Complete standalone vector search and evaluation (1400+ lines)
- resume_eval_core: Lightweight wrapper that imports from other modules (380 lines)

Usage:
    # Import from vector_search_eval (comprehensive standalone)
    from functions import evaluate_resume, evaluate_with_llm
    
    # Or import from resume_eval_core (lightweight, requires other modules)
    from functions.resume_eval_core import evaluate_resume, evaluate_with_llm
"""

# You can import directly from vector_search_eval
from .vector_search_eval import (
    # Main evaluation functions
    evaluate_resume,
    evaluate_with_llm,
    
    # Embedding generation
    generate_embedding,
    
    # Vector search
    find_similar_jobs_local,
    find_similar_jobs_combined,
    find_top_jobs_for_resume,
    
    # Model loading
    load_sbert_model,
    load_trained_word2vec_model,
    load_trained_topic_model,
    get_lsa_100_topics_model,
    
    # Skill extraction
    extract_skills_keywords,
    build_skill_ner,
    extract_skill_entities,
    skill_jaccard_score,
    get_resume_and_job_skills,
    
    # Text processing
    clean_text,
    extract_text_from_pdf,
    simple_tokenize,
    
    # Topic modeling
    get_document_topics,
    compute_topic_similarity,
    compute_topic_score,
    
    # Embeddings computation
    compute_job_embeddings_sbert,
    compute_job_embeddings_w2v,
    get_doc_embedding_w2v,
    
    # Utilities
    check_dependencies,
    print_dependencies,
    MASTER_SKILL_LIST,
)

__all__ = [
    # Main functions
    'evaluate_resume',
    'evaluate_with_llm',
    'generate_embedding',
    'find_similar_jobs_local',
    'find_similar_jobs_combined',
    'find_top_jobs_for_resume',
    
    # Models
    'load_sbert_model',
    'load_trained_word2vec_model',
    'load_trained_topic_model',
    'get_lsa_100_topics_model',
    
    # Skills
    'extract_skills_keywords',
    'build_skill_ner',
    'extract_skill_entities',
    'skill_jaccard_score',
    'get_resume_and_job_skills',
    
    # Text
    'clean_text',
    'extract_text_from_pdf',
    'simple_tokenize',
    
    # Topics
    'get_document_topics',
    'compute_topic_similarity',
    'compute_topic_score',
    
    # Embeddings
    'compute_job_embeddings_sbert',
    'compute_job_embeddings_w2v',
    'get_doc_embedding_w2v',
    
    # Utils
    'check_dependencies',
    'print_dependencies',
    'MASTER_SKILL_LIST',
]

__version__ = '1.0.0'
