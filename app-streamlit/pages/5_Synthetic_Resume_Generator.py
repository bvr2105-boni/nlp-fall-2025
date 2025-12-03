import streamlit as st
import os
import json
import random
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

from components.header import render_header
from utils import initialize_workspace

# Initialize workspace path and imports (for future reuse if needed)
# This also loads the mounted .env file via python-dotenv, so
# OLLAMA_HOST / OLLAMA_API_URL / model env vars should come from there.
initialize_workspace()

try:
    from reportlab.lib.pagesizes import LETTER  # type: ignore
    from reportlab.pdfgen import canvas  # type: ignore
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # type: ignore
    from reportlab.platypus import (  # type: ignore
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        ListFlowable,
        ListItem,
    )
    from reportlab.lib.units import inch  # type: ignore
    REPORTLAB_AVAILABLE = True
    REPORTLAB_ERROR = None
except Exception as e:  # pragma: no cover - optional dependency
    REPORTLAB_AVAILABLE = False
    REPORTLAB_ERROR = e

# Page configuration
st.set_page_config(
    page_title="Synthetic Resume Generator - Ollama",
    page_icon="🧪",
    layout="wide"
)

# Inform about PDF capability after page config
if not REPORTLAB_AVAILABLE:
    # Let the user know that only TXT export is active
    st.caption("PDF export is currently disabled (missing optional dependency `reportlab`).")
# Load global CSS
try:
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles", "app.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

st.title("Synthetic Resume Generator (LLM)")

try:
    import ollama  # type: ignore
except Exception as e:  # pragma: no cover - import-time environment specific
    st.error("⚠️ Ollama Python client is not available.")
    st.info(
        "Install and run Ollama (or ensure the remote Ollama API is reachable). "
        "Example: `pip install ollama` and start the Ollama server."
    )
    st.stop()


def _get_ollama_client() -> "ollama.Client":  # type: ignore[name-defined]
    """
    Construct an Ollama client using an explicit API URL if provided.

    Preference order (all from .env, already loaded via initialize_workspace()):
    1. OLLAMA_API_URL
    2. OLLAMA_HOST
    3. Default http://127.0.0.1:11434
    """
    api_url = (
        os.getenv("OLLAMA_API_URL")
        or os.getenv("OLLAMA_HOST")
        or "http://127.0.0.1:11434"
    )

    # The official python client accepts `host=` to override the base URL
    return ollama.Client(host=api_url)  # type: ignore[attr-defined]


def _normalize_resume_text(text: str) -> str:
    """
    Normalize some common Unicode bullets / separators so the PDF renders cleanly.
    Example: '555·123·4567' or '• Responsibility' -> use simple ASCII equivalents.
    """
    replacements = {
        # Bullets / separators
        "•": "-",
        "●": "-",
        "▪": "-",
        "■": "-",
        "‣": "-",
        "·": "-",
        "• ": "- ",
        # Dashes
        "–": "-",
        "—": "-",
        # Quotes / apostrophes
        "“": "\"",
        "”": "\"",
        "„": "\"",
        "’": "'",
        "‘": "'",
        # Misc spaces
        "\u00A0": " ",  # non-breaking space
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Collapse any doubled hyphens that might appear into a single hyphen with space context
    text = text.replace("--", "-")
    return text


def _get_resume_output_dir() -> str:
    """
    Resolve the directory where synthetic resumes are saved.

    Prefer the main NLP workspace (`workspace/Resume_testing`), but fall back to a
    relative path if the workspace path is not set.
    """
    workspace_path = getattr(st.session_state, "workspace_path", None)
    if workspace_path:
        base_dir = os.path.join(workspace_path, "Resume_testing")
    else:
        # Fallback: relative to project root if workspace path is not set
        base_dir = os.path.join("workspace", "Resume_testing")

    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _save_resume_to_workspace(profile: Dict[str, Any], resume_text: str) -> Optional[str]:
    """
    Save the generated resume as a .txt file under workspace/Resume_testing.

    File name pattern:
      <candidate_name_or_job_title>_<YYYYmmdd_HHMMSS>.txt
    """
    try:
        # Normalize text so PDF/HTML rendering uses simple, clean characters
        resume_text = _normalize_resume_text(resume_text)
        out_dir = _get_resume_output_dir()
        raw_name = profile.get("candidate_name") or profile.get("job_title") or "synthetic_resume"

        # Simple sanitization for file name
        safe_name = "".join(c for c in raw_name if c.isalnum() or c in (" ", "_", "-")).strip()
        if not safe_name:
            safe_name = "synthetic_resume"
        safe_name = safe_name.replace(" ", "_")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}.txt"
        full_path = os.path.join(out_dir, filename)

        header_lines = [
            f"Candidate: {profile.get('candidate_name', 'N/A')}",
            f"Target Title: {profile.get('job_title', 'N/A')}",
            f"Years of Experience: {profile.get('years_experience', 'N/A')}",
            f"Degree: {profile.get('degree', '')} in {profile.get('major', '')}",
            "",
            resume_text,
        ]

        with open(full_path, "w", encoding="utf-8") as f:
            f.write("\n".join(header_lines))

        # 3. Save HTML version (backup / easier to view)
        html_path = os.path.splitext(full_path)[0] + ".html"
        try:
            html_content = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Helvetica, sans-serif; max-width: 800px; margin: 40px auto; line-height: 1.6; }}
                    h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
                    .meta {{ color: #666; margin-bottom: 30px; }}
                    .section {{ margin-top: 25px; white-space: pre-wrap; }}
                </style>
            </head>
            <body>
                <h1>{profile.get('candidate_name', 'Candidate')}</h1>
                <div class="meta">
                    <strong>{profile.get('job_title', 'Job Title')}</strong><br>
                    Experience: {profile.get('years_experience', 'N/A')} years<br>
                    Education: {profile.get('degree', '')} in {profile.get('major', '')}
                </div>
                <div class="section">
                    {resume_text}
                </div>
            </body>
            </html>
            """
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            # st.info(f"📄 HTML version saved to `{html_path}`")
        except Exception as html_e:
            st.warning(f"HTML save failed: {html_e}")

        # 4. Try saving PDF
        if REPORTLAB_AVAILABLE:
            try:
                pdf_path = os.path.splitext(full_path)[0] + ".pdf"
                # Use Platypus for automatic wrapping and pagination
                doc = SimpleDocTemplate(  # type: ignore[name-defined]
                    pdf_path,
                    pagesize=LETTER,  # type: ignore[name-defined]
                    leftMargin=0.75 * inch,  # type: ignore[name-defined]
                    rightMargin=0.75 * inch,  # type: ignore[name-defined]
                    topMargin=0.75 * inch,  # type: ignore[name-defined]
                    bottomMargin=0.75 * inch,  # type: ignore[name-defined]
                )
                styles = getSampleStyleSheet()  # type: ignore[name-defined]
                normal = styles["Normal"]
                heading = styles["Heading2"]

                story = []

                # Header fields
                name_line = f"{profile.get('candidate_name', 'N/A')} — {profile.get('job_title', 'N/A')}"
                story.append(Paragraph(name_line, heading))
                story.append(Paragraph(f"Years of Experience: {profile.get('years_experience', 'N/A')}", normal))
                degree_line = f"{profile.get('degree', '')} in {profile.get('major', '')}"
                story.append(Paragraph(degree_line, normal))
                story.append(Spacer(1, 0.2 * inch))

                # Resume body (plain text -> paragraphs)
                for block in resume_text.split("\n\n"):
                    line = block.strip()
                    if not line:
                        continue
                    # Simple bullet rendering if lines start with "-" (or original bullets)
                    if (
                        "\n- " in block
                        or line.startswith("- ")
                        or "\n• " in block
                        or line.startswith("• ")
                    ):
                        bullets = []
                        for raw in block.split("\n"):
                            raw = raw.strip()
                            if raw.startswith("- ") or raw.startswith("• "):
                                bullets.append(Paragraph(raw[2:].strip(), normal))
                        if bullets:
                            story.append(ListFlowable(  # type: ignore[name-defined]
                                [ListItem(b) for b in bullets],  # type: ignore[name-defined]
                                bulletType="bullet",
                                leftIndent=12,
                            ))
                            story.append(Spacer(1, 0.1 * inch))
                    else:
                        story.append(Paragraph(line.replace("\n", "<br/>"), normal))
                        story.append(Spacer(1, 0.08 * inch))

                doc.build(story)
                # st.info(f"PDF version also saved to `{pdf_path}`")

                # If PDF was successfully created, remove the intermediate TXT/HTML files
                try:
                    if os.path.exists(full_path):
                        os.remove(full_path)
                    if os.path.exists(html_path):
                        os.remove(html_path)
                except Exception as cleanup_err:
                    # Non-fatal; just let the user know extra files remain
                    st.warning(f"PDF created, but could not remove TXT/HTML: {cleanup_err}")
            except Exception as pdf_err:
                # Make this error visible!
                st.error(f"CRITICAL PDF SAVE ERROR: {pdf_err}")
                import traceback
                st.code(traceback.format_exc())

        return full_path
    except Exception as e:
        st.error(f"Error saving resume to workspace: {e}")
        return None


def _sample_random_profile_params(
    base_job_title: str,
    base_years: int,
    base_degree: str,
    base_major: str,
    base_region: str,
) -> Tuple[str, int, str, str, str]:
    """
    Sample random, but reasonable, profile parameters for batch generation.

    We keep samples near the user-specified values but introduce variety.
    """
    job_title_choices = [
        base_job_title or "Data Scientist",
        "Machine Learning Engineer",
        "Data Analyst",
        "NLP Engineer",
        "MLOps Engineer",
        "Analytics Engineer",
    ]

    degree_choices = [
        "Bachelor's",
        "Master's",
        "PhD",
        "Associate",
        "Bootcamp / Diploma",
        "No formal degree",
    ]

    major_choices = [
        base_major or "Computer Science",
        "Data Science",
        "Statistics",
        "Mathematics",
        "Electrical Engineering",
        "Information Systems",
    ]

    region_choices = [
        base_region or "United States",
        "North America",
        "Europe",
        "Asia-Pacific",
        "Remote / Global",
    ]

    # Years of experience sampled around the base value
    min_years = max(0, base_years - 3)
    max_years = min(40, base_years + 5) if base_years > 0 else 10
    rand_years = random.randint(min_years, max_years)

    rand_job_title = random.choice(job_title_choices)
    rand_degree = random.choice(degree_choices)
    rand_major = random.choice(major_choices)
    rand_region = random.choice(region_choices)

    return rand_job_title, rand_years, rand_degree, rand_major, rand_region


def _generate_and_save_single_resume(
    model_name: str,
    job_title: str,
    years_experience: int,
    degree_level: str,
    major: str,
    region: str,
) -> Optional[Dict[str, Any]]:
    """
    Helper to generate one profile + resume and save to workspace.

    Returns a dict with profile, resume, and path if successful.
    """
    profile = generate_candidate_profile(
        model=model_name.strip(),
        job_title=job_title.strip(),
        years_experience=int(years_experience),
        degree_level=degree_level,
        major=major.strip() or "Undeclared",
        region=region.strip() or "Global",
    )

    if profile is None:
        return None

    resume = generate_resume_from_profile(model_name.strip(), profile)
    if resume is None:
        return None

    saved_path = _save_resume_to_workspace(profile, resume)
    return {
        "profile": profile,
        "resume": resume,
        "saved_path": saved_path,
    }


# --- Helper functions for Ollama calls ---
def generate_candidate_profile(
    model: str,
    job_title: str,
    years_experience: int,
    degree_level: str,
    major: str,
    region: str,
) -> Optional[Dict[str, Any]]:
    """Call Ollama to generate a structured candidate profile.

    The model is instructed to return STRICT JSON with fields that we can parse,
    and to keep work history and education logically consistent with years of
    experience and major.
    """

    # Add randomization seed to encourage name diversity
    random_seed = random.randint(1, 10000)
    
    system_prompt = (
        "You are an expert CV writer and career coach. "
        "You generate realistic candidate profiles for synthetic resumes. "
        "CRITICAL: Use a DIFFERENT, unique name for EVERY candidate. "
        "Vary names across different cultural backgrounds, ethnicities, and regions. "
        "Do NOT reuse the same first name or last name combination. "
        "Use diverse names like: Maria Rodriguez, James Chen, Priya Sharma, Ahmed Hassan, "
        "Emma Thompson, Carlos Mendez, Yuki Tanaka, Fatima Al-Rashid, etc. "
        f"Random seed: {random_seed}"
    )

    user_prompt = f"""
Generate a realistic candidate profile for a synthetic resume.

CONSTRAINTS:
- Target job title: {job_title}
- Total professional experience: {years_experience} years
- Highest degree: {degree_level}
- Major / field of study: {major}
- Region / market (for company names): {region or 'any global'}

RULES:
1. Work history MUST be consistent with the requested years of experience.
   - The sum of years across positions should be roughly {years_experience} years.
   - Use realistic timelines with no impossible overlaps.
2. Education MUST match the requested major/field and degree level.
3. Previous positions should form a plausible career progression toward the target job title.
4. NAME REQUIREMENT: Generate a UNIQUE, diverse name that you have NOT used before.
   - Choose from different cultural backgrounds (Asian, Hispanic, Middle Eastern, European, African, etc.)
   - Vary both first names AND last names significantly
   - Examples of diverse names: Keiko Yamamoto, Marcus Johnson, Sofia Martinez, Arjun Patel, 
     Leila Al-Mansouri, David Kim, Elena Petrov, etc.
   - Do NOT use common repetitive names like "Aisha Patel" repeatedly
5. Use realistic but fully synthetic names for companies.

Return ONLY a single JSON object with this exact structure and no extra text:
{{
  "candidate_name": string,
  "job_title": string,
  "years_experience": integer,
  "degree": string,
  "major": string,
  "summary": string,
  "locations": [string, ...],
  "skills": [string, ...],
  "positions": [
    {{
      "title": string,
      "company": string,
      "location": string,
      "start_year": integer,
      "end_year": integer,
      "responsibilities": [string, ...]
    }},
    ...
  ],
  "education": [
    {{
      "degree": string,
      "major": string,
      "institution": string,
      "location": string,
      "graduation_year": integer
    }}
  ]
}}
""".strip()

    try:
        client = _get_ollama_client()
        # Use structured output so we can safely parse JSON
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format="json",  # Ollama will format the content as JSON
        )
        raw_content = response["message"]["content"]
        profile = json.loads(raw_content)
        return profile
    except Exception as e:
        st.error(f"Error generating candidate profile: {e}")
        return None


def generate_resume_from_profile(model: str, profile: Dict[str, Any]) -> Optional[str]:
    """Call Ollama again to generate a full resume based on the profile.

    We embed the structured profile and explicitly require all positions and
    education in the resume to match the profile (titles, years, major, etc.).
    """

    profile_json_pretty = json.dumps(profile, indent=2)

    system_prompt = (
        "You are an expert resume writer. You create ATS-friendly, well-structured "
        "resumes based on a provided candidate profile."
    )

    user_prompt = f"""
Using the candidate profile below, write a complete, one- to two-page resume.

CANDIDATE PROFILE (SOURCE OF TRUTH):
{profile_json_pretty}

STRICT CONSISTENCY RULES:
1. All job titles, companies, locations, and years MUST match the positions array.
2. Total years of experience must stay consistent with "years_experience".
3. Education section MUST match the education array (degree, major, institution, year).
4. Do NOT invent extra degrees or jobs that are not in the profile.
5. You may refine wording and bullet points, but not change the factual timeline.

FORMAT REQUIREMENTS:
- Start with the candidate's NAME on the first line in ALL CAPS (e.g., "MAYA TRAN").
- On the second line, put the target JOB TITLE.
- On the third line, put a single CONTACT line in this style (ASCII only, no special bullets):
  "San Francisco, CA - (555) 123-4567 - email@example.com - LinkedIn: linkedin.com/in/example"
- Use section headings in ALL CAPS (e.g., "SUMMARY", "SKILLS", "EXPERIENCE", "EDUCATION").
- Use ONLY simple ASCII characters. Do NOT use special bullets such as "■", "▪", "●", "‣", or "·".
- For bullet points, always start lines with a plain hyphen and a space, like "- Led a team of 5 engineers ...".
- Keep lines reasonably wrapped (no extremely long single lines) and write in polished, professional English
  tailored to the target job title.
""".strip()

    try:
        client = _get_ollama_client()
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        resume_text = response["message"]["content"]
        return resume_text
    except Exception as e:
        st.error(f"Error generating resume: {e}")
        return None


# --- UI ---

st.markdown(
    """
### Overview
Create a **synthetic, realistic resume** using a local **LLM** model in two stages:
- **Stage 1**: Generate a consistent candidate profile (name, target job, timeline, education).
- **Stage 2**: Generate a full resume whose positions and education strictly follow that profile.
"""
)

# Internal model configuration (hidden from UI)
# Prefer model name from environment (.env), with a sensible default.
model_name = (
    os.getenv("OLLAMA_MODEL")
    or os.getenv("OLLAMA_DEFAULT_MODEL")
    or "gpt-oss:20b"
)

# Input form for high-level parameters
st.markdown("---")
st.markdown("### Define Target Profile")

col_job, col_years = st.columns([2, 1])
with col_job:
    job_title = st.text_input("Target job title", value="Data Scientist")
with col_years:
    years_experience = st.slider("Years of experience", min_value=0, max_value=40, value=5)

col_degree, col_major = st.columns(2)
with col_degree:
    degree_level = st.selectbox(
        "Highest degree",
        [
            "Bachelor's",
            "Master's",
            "PhD",
            "Associate",
            "Bootcamp / Diploma",
            "No formal degree",
        ],
        index=0,
    )
with col_major:
    major = st.text_input("Major / field of study", value="Computer Science")

region = st.text_input(
    "Region / market (for realistic companies)",
    value="United States",
)

st.markdown("---")
st.markdown("### Batch Generation Settings")

max_resumes = st.slider(
    "Max resumes this run",
    min_value=1,
    max_value=100,
    value=1,
    help="How many synthetic resumes to generate and save in one click.",
)

use_random_params = st.checkbox(
    "Randomize profile parameters for each resume",
    value=False,
    help="If checked, job title, years, degree, major, and region are sampled automatically for each resume.",
)

if "synthetic_profile" not in st.session_state:
    st.session_state.synthetic_profile = None
if "synthetic_resume" not in st.session_state:
    st.session_state.synthetic_resume = None

col_gen1, col_gen2 = st.columns([2, 1])

with col_gen1:
    generate_both = st.button(
        "✨ Generate Candidate & Resume",
        type="primary",
        use_container_width=True,
    )

with col_gen2:
    regenerate_resume_only = st.button(
        "Regenerate Resume",
        use_container_width=True,
    )

# --- Generation logic ---

if generate_both:
    if max_resumes == 1 and not use_random_params:
        # Original single-run behavior using explicit parameters
        if not job_title.strip():
            st.error("Please provide a target job title.")
        else:
            with st.spinner("Generating candidate profile with LLM..."):
                profile = generate_candidate_profile(
                    model=model_name.strip(),
                    job_title=job_title.strip(),
                    years_experience=int(years_experience),
                    degree_level=degree_level,
                    major=major.strip() or "Undeclared",
                    region=region.strip() or "Global",
                )

            if profile is not None:
                st.session_state.synthetic_profile = profile
                st.success("✅ Candidate profile generated.")

                with st.spinner("Generating resume from profile..."):
                    resume = generate_resume_from_profile(model_name.strip(), profile)

                if resume is not None:
                    st.session_state.synthetic_resume = resume

                    # Save to workspace/Resume_testing
                    saved_path = _save_resume_to_workspace(profile, resume)
                    if saved_path:
                        st.success(f"✅ Resume generated and saved to `{saved_path}`")
                        pdf_path = os.path.splitext(saved_path)[0] + ".pdf"
                        if os.path.exists(pdf_path):
                            st.success(f"📄 PDF saved to `{pdf_path}`")
                            try:
                                with open(pdf_path, "rb") as f:
                                    st.download_button(
                                        "Download PDF",
                                        f.read(),
                                        file_name=os.path.basename(pdf_path),
                                        mime="application/pdf",
                                        use_container_width=True,
                                    )
                            except Exception:
                                pass
                    else:
                        st.success("✅ Resume generated (but could not be saved to workspace).")
    else:
        # Batch mode (single or multiple resumes), optionally randomizing parameters
        results = []
        with st.spinner(f"Generating up to {max_resumes} resumes with Ollama..."):
            for i in range(max_resumes):
                if use_random_params:
                    jt, yrs, deg, maj, reg = _sample_random_profile_params(
                        base_job_title=job_title,
                        base_years=int(years_experience),
                        base_degree=degree_level,
                        base_major=major,
                        base_region=region,
                    )
                else:
                    jt, yrs, deg, maj, reg = (
                        job_title,
                        int(years_experience),
                        degree_level,
                        major,
                        region,
                    )

                single = _generate_and_save_single_resume(
                    model_name=model_name,
                    job_title=jt,
                    years_experience=yrs,
                    degree_level=deg,
                    major=maj,
                    region=reg,
                )
                if single:
                    results.append(single)

        if results:
            # Keep the last generated profile/resume on screen
            last = results[-1]
            st.session_state.synthetic_profile = last["profile"]
            st.session_state.synthetic_resume = last["resume"]

            saved_paths = [r["saved_path"] for r in results if r.get("saved_path")]
            count = len(results)
            st.success(f"✅ Generated {count} resume(s).")

            # if saved_paths:
            #     st.markdown("**Saved files (workspace/Resume_testing):**")
            #     for p in saved_paths:
            #         st.code(p, language="bash")
        else:
            st.error("No resumes were successfully generated. Please check the model and try again.")

if regenerate_resume_only and st.session_state.get("synthetic_profile") is not None:
    with st.spinner("Regenerating resume from existing profile..."):
        resume = generate_resume_from_profile(
            model_name.strip(), st.session_state.synthetic_profile
        )
    if resume is not None:
        st.session_state.synthetic_resume = resume

        # Also save regenerated resume
        saved_path = _save_resume_to_workspace(st.session_state.synthetic_profile, resume)
        if saved_path:
            st.success(f"✅ Resume regenerated and saved to `{saved_path}`")
            pdf_path = os.path.splitext(saved_path)[0] + ".pdf"
            if os.path.exists(pdf_path):
                st.success(f"📄 PDF saved to `{pdf_path}`")
                try:
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "Download PDF",
                            f.read(),
                            file_name=os.path.basename(pdf_path),
                            mime="application/pdf",
                            use_container_width=True,
                        )
                except Exception:
                    pass
        else:
            st.success("✅ Resume regenerated (but could not be saved to workspace).")


# --- Display results ---

profile = st.session_state.get("synthetic_profile")
resume = st.session_state.get("synthetic_resume")

if profile is not None:
    st.markdown("---")
    st.markdown("### Generated Candidate Profile")

    header_cols = st.columns([2, 2, 1, 2])
    with header_cols[0]:
        st.markdown(f"**Name:** {profile.get('candidate_name', 'N/A')}")
    with header_cols[1]:
        st.markdown(f"**Target Title:** {profile.get('job_title', job_title)}")
    with header_cols[2]:
        st.markdown(f"**Experience:** {profile.get('years_experience', years_experience)} yrs")
    with header_cols[3]:
        degree_str = profile.get('degree', degree_level)
        major_str = profile.get('major', major)
        st.markdown(f"**Education:** {degree_str} in {major_str}")

    if profile.get("summary"):
        st.markdown("**Summary**")
        st.write(profile["summary"])

    col_left, col_right = st.columns(2)

    with col_left:
        positions = profile.get("positions", [])
        st.markdown("**Experience (Resume-style)**")
        if positions:
            for pos in positions:
                title = pos.get("title", "")
                company = pos.get("company", "")
                loc = pos.get("location", "")
                start = pos.get("start_year", "?")
                end = pos.get("end_year", "Present")

                st.markdown(f"**{title}**, {company} — {loc}")
                st.caption(f"{start} – {end}")

                responsibilities = pos.get("responsibilities", [])
                if responsibilities:
                    for r in responsibilities:
                        st.markdown(f"- {r}")
                st.markdown("")  # spacing between roles
        else:
            st.write("No positions returned by model.")

    with col_right:
        education = profile.get("education", [])
        st.markdown("**Education**")
        if education:
            for edu in education:
                deg = edu.get("degree", "")
                maj = edu.get("major", "")
                inst = edu.get("institution", "")
                loc = edu.get("location", "")
                grad = edu.get("graduation_year", "")
                st.markdown(f"- **{deg} in {maj}**, {inst} ({grad}) — {loc}")
        else:
            st.write("No education entries returned by model.")

        skills = profile.get("skills", [])
        if skills:
            st.markdown("**Key Skills**")
            st.write(", ".join(skills))

    with st.expander("🔍 Raw Profile JSON"):
        st.json(profile)

if resume is not None:
    st.markdown("---")
    st.markdown("### Generated Resume")
    st.text_area(
        "Synthetic resume text",
        value=resume,
        height=500,
    )

# Footer
st.markdown("---")
st.caption(
    "Generated locally using Ollama. All profiles and resumes are synthetic and "
    "intended for testing and demo purposes only."
)

with st.expander("🛠️ Debug Info"):
    st.write(f"**ReportLab Available:** `{REPORTLAB_AVAILABLE}`")
    if not REPORTLAB_AVAILABLE:
        st.error(f"Import Error: {REPORTLAB_ERROR}")
        st.info("Try rebuilding: `docker compose build nlp-streamlit-app`")
    
    if "synthetic_resume" in st.session_state and st.session_state.synthetic_resume:
         st.write("**Last Resume Length:**", len(st.session_state.synthetic_resume))
