import streamlit as st
import os
from components.header import render_header

# Page configuration
st.set_page_config(
    page_title="Meet Our Team",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load global CSS
try:
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles", "app.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# Header section
st.title("Our Team")

st.markdown("""
<div style="margin-bottom: 2rem;">
    <p style="color: #6b7280; font-size: 1.1rem; line-height: 1.6;">
        Meet the talented team behind this AI-powered job intelligence platform. <br/>
        We're passionate about using advanced technology to transform career discovery and workforce analytics.
    </p>
</div>
""", unsafe_allow_html=True)

# Mobile responsive tweaks
st.markdown("""
<style>
/* Stack columns and scale images on small screens */
@media (max-width: 768px) {
  /* Reduce side padding for more space */
  .block-container {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
  }
  /* Make each Streamlit column span full width */
  div[data-testid="column"] {
    width: 100% !important;
    flex: 1 0 100% !important;
  }
  /* Tighten gaps between stacked items */
  div[data-testid="stHorizontalBlock"] {
    gap: 0.75rem !important;
  }
  /* Ensure images scale to container width */
  [data-testid="stImage"] img {
    width: 100% !important;
    height: auto !important;
    border-radius: 8px;
  }
  /* Center captions under images */
  [data-testid="stImage"] figcaption {
    text-align: center;
  }
}
/* Slightly larger gaps on wider screens */
@media (min-width: 769px) {
  div[data-testid="stHorizontalBlock"] {
    gap: 1.25rem !important;
  }
}
</style>
""", unsafe_allow_html=True)

# Display team photos
images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")
team_members = [
    ("Boni Vasius Rosen", "boni.jpg"),
    ("Minkyung (Ginny) Kim", "ginny.jpg"),
    ("Kas Kiatsukasem", "kas.jpeg"),
    ("Kibaek Kim", "kibaek.jpg"),
    ("Suchakrey (Philip) Nitisanon", "philip.jpg"),
]

cols = st.columns(len(team_members))
for col, (name, filename) in zip(cols, team_members):
    img_path = os.path.join(images_dir, filename)
    try:
        col.image(img_path, caption=name, use_container_width=True)
    except Exception:
        col.info(f"{name}'s photo coming soon...")