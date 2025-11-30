import streamlit as st
import os

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
st.markdown("""
<div class="main-header">
    <h1>Meet Our Team</h1>
    <p>NLP Team Members</p>
</div>
""", unsafe_allow_html=True)


# Display team image
try:
    st.image("images/Team.png", caption="Capstone Fall 2025 Team")
except:
    st.write("Team image not found. Please ensure 'Team.png' is in the images/ folder.")

        