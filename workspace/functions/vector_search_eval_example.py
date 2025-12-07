"""
Example Usage of Vector Search and Evaluation Functions
========================================================

This script demonstrates how to use the vector_search_eval module
for resume evaluation and job matching without Streamlit.

Author: NLP Fall 2025 Project
Date: December 2025
"""

import pandas as pd
import numpy as np
from vector_search_eval import (
    # Main evaluation function
    evaluate_resume,
    
    # Individual component functions
    generate_embedding,
    find_similar_jobs_local,
    find_similar_jobs_combined,
    evaluate_with_llm,
    
    # Model loading functions
    load_sbert_model,
    load_trained_word2vec_model,
    load_trained_topic_model,
    
    # Skill extraction
    extract_skills_keywords,
    build_skill_ner,
    extract_skill_entities,
    
    # Text processing
    clean_text,
    extract_text_from_pdf,
    
    # Utilities
    check_dependencies,
    print_dependencies,
    MASTER_SKILL_LIST,
)


def example_1_check_dependencies():
    """Example 1: Check which dependencies are available"""
    print("=" * 60)
    print("Example 1: Checking Dependencies")
    print("=" * 60)
    print_dependencies()
    print()


def example_2_simple_embedding():
    """Example 2: Generate embeddings for text"""
    print("=" * 60)
    print("Example 2: Generate Embeddings")
    print("=" * 60)
    
    sample_text = "Python developer with 5 years of experience in machine learning and data science"
    
    # Generate SBERT embedding
    print("Generating SBERT embedding...")
    sbert_emb = generate_embedding(sample_text, method="sbert")
    if sbert_emb is not None:
        print(f"✅ SBERT embedding shape: {sbert_emb.shape}")
    else:
        print("❌ SBERT embedding failed")
    
    # Generate Word2Vec embedding
    print("\nGenerating Word2Vec embedding...")
    w2v_emb = generate_embedding(sample_text, method="word2vec", models_dir="../models")
    if w2v_emb is not None:
        print(f"✅ Word2Vec embedding shape: {w2v_emb.shape}")
    else:
        print("❌ Word2Vec embedding failed (model may not be trained yet)")
    
    print()


def example_3_skill_extraction():
    """Example 3: Extract skills from text"""
    print("=" * 60)
    print("Example 3: Skill Extraction")
    print("=" * 60)
    
    resume_text = """
    Software Engineer with expertise in Python, Java, and JavaScript.
    Strong background in machine learning, deep learning, and NLP.
    Experience with AWS, Docker, and Kubernetes.
    Proficient in SQL, PostgreSQL, and MongoDB.
    """
    
    # Method 1: Keyword-based extraction (always works)
    print("Method 1: Keyword-based extraction")
    skills_keywords = extract_skills_keywords(resume_text, MASTER_SKILL_LIST)
    print(f"Found {len(skills_keywords)} skills: {skills_keywords}")
    
    # Method 2: NER-based extraction (requires spaCy)
    print("\nMethod 2: NER-based extraction (requires spaCy)")
    matcher = build_skill_ner(MASTER_SKILL_LIST)
    if matcher:
        skills_ner = extract_skill_entities(resume_text, matcher)
        print(f"Found {len(skills_ner)} skills: {skills_ner}")
    else:
        print("❌ spaCy not available, using keyword method as fallback")
    
    print()


def example_4_vector_search():
    """Example 4: Vector search for similar jobs"""
    print("=" * 60)
    print("Example 4: Vector Search for Similar Jobs")
    print("=" * 60)
    
    # Sample data
    resume_text = "Python developer with machine learning experience"
    
    job_data = {
        'id': [1, 2, 3, 4, 5],
        'title': [
            'Python Developer',
            'Machine Learning Engineer',
            'Java Developer',
            'Data Scientist',
            'Frontend Developer'
        ],
        'company': ['Company A', 'Company B', 'Company C', 'Company D', 'Company E'],
        'text': [
            'Looking for Python developer with backend experience',
            'ML engineer needed for deep learning projects using Python',
            'Senior Java developer for enterprise applications',
            'Data scientist with Python and machine learning expertise',
            'React and JavaScript frontend developer'
        ]
    }
    df = pd.DataFrame(job_data)
    
    print("Resume:", resume_text)
    print(f"\nSearching through {len(df)} jobs...")
    
    # Generate embeddings
    resume_emb = generate_embedding(resume_text, method="sbert")
    if resume_emb is None:
        print("❌ Could not generate embeddings")
        return
    
    job_texts = df['text'].tolist()
    from vector_search_eval import compute_job_embeddings_sbert
    job_embeddings = compute_job_embeddings_sbert(job_texts)
    
    if job_embeddings is None:
        print("❌ Could not generate job embeddings")
        return
    
    # Find similar jobs
    results = find_similar_jobs_local(resume_emb, job_embeddings, df, top_k=3)
    
    print(f"\nTop {len(results)} matching jobs:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']} at {result['company']}")
        print(f"   Similarity: {result['similarity']:.3f}")
        print(f"   Description: {result['text'][:80]}...")
    
    print()


def example_5_complete_evaluation():
    """Example 5: Complete resume evaluation with all scores"""
    print("=" * 60)
    print("Example 5: Complete Resume Evaluation")
    print("=" * 60)
    
    # Sample resume
    resume_text = """
    Senior Data Scientist
    
    Skills:
    - Python, R, SQL
    - Machine Learning, Deep Learning, NLP
    - TensorFlow, PyTorch, Scikit-learn
    - AWS, Docker
    - Pandas, NumPy, Matplotlib
    
    Experience:
    5+ years in machine learning and data science
    Developed NLP models for text classification
    Built recommendation systems using collaborative filtering
    """
    
    # Sample jobs
    job_data = {
        'id': [1, 2, 3],
        'title': [
            'Senior Data Scientist',
            'Machine Learning Engineer',
            'Java Backend Developer'
        ],
        'company': ['TechCorp', 'AI Startup', 'Enterprise Inc'],
        'text': [
            'Senior Data Scientist needed. Required: Python, ML, NLP, AWS. 5+ years experience.',
            'ML Engineer position. Python, TensorFlow, PyTorch required. Computer vision focus.',
            'Java developer for backend systems. Spring Boot, Microservices, REST APIs.'
        ]
    }
    df = pd.DataFrame(job_data)
    job_texts = df['text'].tolist()
    
    print("Evaluating resume against jobs...\n")
    
    # Run evaluation (without LLM for faster results)
    results = evaluate_resume(
        resume_text=resume_text,
        job_texts=job_texts,
        df=df,
        skill_list=MASTER_SKILL_LIST,
        top_k=3,
        embedding_type="sbert",
        run_llm=False,  # Set to True to enable LLM evaluation
        models_dir="../models"
    )
    
    # Display results
    for result in results:
        print(f"Rank {result['job_rank']}: {result['job_title']} at {result['job_company']}")
        print(f"  Final Score: {result['final_score']:.3f}")
        print(f"  - Skill Score: {result['skill_score']:.3f}")
        print(f"  - Semantic Score: {result['semantic_score']:.3f}")
        print(f"  - Topic Score: {result['topic_score']:.3f}")
        print(f"  Matching Skills: {', '.join(result['matching_skills']) if result['matching_skills'] else 'None'}")
        print()
    
    print()


def example_6_llm_evaluation():
    """Example 6: LLM-based evaluation (requires Ollama)"""
    print("=" * 60)
    print("Example 6: LLM Evaluation (requires Ollama)")
    print("=" * 60)
    
    resume_text = "Python developer with 3 years of web development experience"
    
    job = {
        'title': 'Senior Python Developer',
        'company': 'Tech Company',
        'text': 'Looking for Senior Python Developer with 5+ years experience in Django and Flask'
    }
    
    print("Evaluating with LLM...\n")
    
    # Run LLM evaluation
    result = evaluate_with_llm(
        resume_text=resume_text,
        job=job,
        model_name=None,  # Will use default from env or "gpt-oss:20b"
        api_url=None,     # Will use default from env or "http://127.0.0.1:11434"
    )
    
    if result['llm_error']:
        print(f"❌ LLM Evaluation Error: {result['llm_error']}")
        print("   (Make sure Ollama is installed and running)")
    else:
        print(f"Match: {'✅ Yes' if result['llm_match'] else '❌ No' if result['llm_match'] is not None else '⚠️ Unknown'}")
        print(f"Reasoning: {result['llm_reasoning']}")
        
        if result['llm_recommendations']:
            print("\nRecommendations:")
            for rec in result['llm_recommendations']:
                print(f"  - {rec}")
        
        if result['linkedin_keywords']:
            print(f"\nLinkedIn Keywords: {', '.join(result['linkedin_keywords'])}")
    
    print()


def example_7_combined_search():
    """Example 7: Combined search with skills, semantic, and topic scoring"""
    print("=" * 60)
    print("Example 7: Combined Search (Skills + Semantic + Topic)")
    print("=" * 60)
    
    resume_text = "Data scientist with Python, machine learning, and NLP experience"
    
    job_data = {
        'id': [1, 2, 3],
        'title': ['Data Scientist', 'ML Engineer', 'Software Engineer'],
        'company': ['A', 'B', 'C'],
        'text': [
            'Data scientist role requiring Python, ML, and statistical analysis',
            'Machine learning engineer with deep learning and computer vision',
            'Software engineer for full stack web development'
        ]
    }
    df = pd.DataFrame(job_data)
    job_texts = df['text'].tolist()
    
    print("Using combined scoring with custom weights...")
    print("Weights: Skills=0.5, Semantic=0.3, Topic=0.2\n")
    
    # Custom weights
    weights = {
        'skills': 0.5,
        'semantic': 0.3,
        'topic': 0.2
    }
    
    results = find_similar_jobs_combined(
        query_text=resume_text,
        job_texts=job_texts,
        df=df,
        top_k=3,
        weights=weights,
        skill_list=MASTER_SKILL_LIST,
        models_dir="../models"
    )
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} at {result['company']}")
        print(f"   Combined: {result['similarity']:.3f}")
        print(f"   Skills: {result['skills_similarity']:.3f}, "
              f"Semantic: {result['semantic_similarity']:.3f}, "
              f"Topic: {result['topic_similarity']:.3f}")
        print()


def example_8_pdf_processing():
    """Example 8: Extract text from PDF resume"""
    print("=" * 60)
    print("Example 8: PDF Processing")
    print("=" * 60)
    
    pdf_path = "../Resume_testing/sample_resume.pdf"  # Update with actual path
    
    print(f"Attempting to extract text from: {pdf_path}")
    
    text = extract_text_from_pdf(pdf_path)
    
    if text:
        print(f"✅ Extracted {len(text)} characters")
        print(f"\nFirst 200 characters:")
        print(text[:200])
        print("...")
    else:
        print("❌ Could not extract text (file may not exist or pypdf not available)")
    
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Vector Search and Evaluation Examples")
    print("=" * 60 + "\n")
    
    # Run examples
    example_1_check_dependencies()
    example_2_simple_embedding()
    example_3_skill_extraction()
    example_4_vector_search()
    example_5_complete_evaluation()
    
    # Optional examples (comment out if not needed)
    # example_6_llm_evaluation()  # Requires Ollama
    # example_7_combined_search()   # Requires trained models
    # example_8_pdf_processing()    # Requires PDF file
    
    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

