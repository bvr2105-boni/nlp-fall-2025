"""
Lightweight resume evaluation helpers without Streamlit dependencies.

These utilities mirror the logic used in the Streamlit pages but avoid
`st.*` usage so they can be imported from notebooks or scripts.
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional imports; the module still works with graceful degradation.
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except Exception:  # noqa: BLE001
    PYPDF_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:  # noqa: BLE001
    OLLAMA_AVAILABLE = False

# Project helpers that do not depend on Streamlit
try:
    from functions.database import find_similar_jobs_vector
except Exception:  # noqa: BLE001
    find_similar_jobs_vector = None


def clean_text(text: str) -> str:
    """Basic text cleaning used for resumes and job descriptions."""
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
    """Extract raw text from a PDF file."""
    if not PYPDF_AVAILABLE:
        return None
    try:
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip()
    except Exception:  # noqa: BLE001
        return None


def extract_skills_keywords(text: str, skill_list: List[str]) -> List[str]:
    """Keyword-only skill extraction (case-insensitive, multi-word aware)."""
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
    """Jaccard similarity of skill sets."""
    r, j = set(resume_skills), set(job_skills)
    union = r | j
    if not union:
        return 0.0
    return len(r & j) / len(union)


def _lazy_import_nlp_models():
    """Import selected helpers from nlp_models lazily to avoid hard deps."""
    try:
        from functions import nlp_models as nm
    except Exception:  # noqa: BLE001
        return None
    return nm


def generate_embedding(text: str, method: str = "sbert"):
    """Generate embedding using local models if available."""
    nm = _lazy_import_nlp_models()
    if nm is None:
        return None
    try:
        return nm.generate_local_embedding(text, method=method)
    except Exception:  # noqa: BLE001
        return None


def compute_topic_score(resume_text: str, job_text: str, nm=None) -> float:
    """Compute topic similarity using cached LSA model when available."""
    if nm is None:
        nm = _lazy_import_nlp_models()
    if nm is None:
        return 0.0
    try:
        topic_model = nm.get_lsa_100_topics_model()
        if topic_model is None:
            return 0.0
        r_topics = nm.get_document_topics(resume_text, topic_model)
        j_topics = nm.get_document_topics(job_text, topic_model)
        if r_topics is None or j_topics is None:
            return 0.0
        score = nm.compute_topic_similarity(r_topics, j_topics)
        return float(np.clip(score, 0.0, 1.0))
    except Exception:  # noqa: BLE001
        return 0.0


def get_resume_and_job_skills(
    resume_text: str, job_text: str, skill_list: List[str], nm=None
) -> Tuple[List[str], List[str]]:
    """Extract skills using matcher if available, else keyword fallback."""
    if nm is None:
        nm = _lazy_import_nlp_models()
    matcher = None
    if nm and hasattr(nm, "build_skill_ner"):
        try:
            matcher = nm.build_skill_ner(skill_list)
        except Exception:  # noqa: BLE001
            matcher = None

    if matcher and nm:
        try:
            resume_skills = nm.extract_skill_entities(resume_text, matcher)
            job_skills = nm.extract_skill_entities(job_text, matcher)
        except Exception:  # noqa: BLE001
            resume_skills = extract_skills_keywords(resume_text, skill_list)
            job_skills = extract_skills_keywords(job_text, skill_list)
    else:
        resume_skills = extract_skills_keywords(resume_text, skill_list)
        job_skills = extract_skills_keywords(job_text, skill_list)
    return resume_skills, job_skills


def find_top_jobs_for_resume(
    resume_text: str,
    skill_list: List[str],
    top_k: int = 3,
    embedding_type: str = "sbert",
) -> List[Dict]:
    """Vector search + scoring against jobs table (pgvector)."""
    if find_similar_jobs_vector is None:
        return []
    resume_emb = generate_embedding(resume_text, method=embedding_type)
    if resume_emb is None:
        return []
    try:
        matches = find_similar_jobs_vector(
            resume_emb.tolist(), embedding_type=embedding_type, top_k=top_k * 2
        )
    except Exception:  # noqa: BLE001
        return []
    if not matches:
        return []

    nm = _lazy_import_nlp_models()
    resume_skills_kw = extract_skills_keywords(resume_text, skill_list)
    enriched = []
    for job in matches:
        job_text = job.get("text", "") or ""
        resume_skills, job_skills = get_resume_and_job_skills(
            resume_text, job_text, skill_list, nm=nm
        )
        if not resume_skills:
            resume_skills = resume_skills_kw
        skill_score = skill_jaccard_score(resume_skills, job_skills)
        semantic_score = job.get("similarity", 0.0)
        topic_score = compute_topic_score(resume_text, job_text, nm=nm)
        if topic_score == 0.0:
            topic_score = semantic_score
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


def evaluate_with_llm(
    resume_text: str,
    job: Dict,
    model_name: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Dict:
    """LLM Yes/No + recommendations using Ollama if available."""
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
    except Exception as e:  # noqa: BLE001
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
    except Exception as e:  # noqa: BLE001
        return {
            "llm_match": None,
            "llm_reasoning": None,
            "llm_recommendations": None,
            "linkedin_keywords": None,
            "llm_error": str(e),
        }

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


def evaluate_resume(
    resume_text: str,
    skill_list: List[str],
    top_k: int = 3,
    embedding_type: str = "sbert",
    run_llm: bool = False,
    llm_model: Optional[str] = None,
    llm_api_url: Optional[str] = None,
) -> List[Dict]:
    """End-to-end evaluation pipeline without Streamlit."""
    resume_text = clean_text(resume_text)
    jobs = find_top_jobs_for_resume(
        resume_text, skill_list=skill_list, top_k=top_k, embedding_type=embedding_type
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

