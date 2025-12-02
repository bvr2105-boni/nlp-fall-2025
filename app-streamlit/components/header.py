import streamlit as st
import os
from typing import Optional


def load_css() -> None:
    """
    Load global CSS from styles/app.css. Safe to call multiple times.
    """
    try:
        css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles", "app.css")
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        # Best-effort styling; don't block the app if CSS is missing
        pass


def setup_page(
    page_title: Optional[str] = None,
    page_icon: str = "💼",
    initial_sidebar_state: str = "expanded",
    layout: str = "wide",
) -> None:
    """
    Standardize page config + CSS + header for a consistent, commercial-grade look.
    Safe to import and call from any page.
    """
    try:
        st.set_page_config(
            layout=layout,
            page_title=page_title or "LinkedIn Job Analysis Platform",
            page_icon=page_icon,
            initial_sidebar_state=initial_sidebar_state,
        )
    except Exception:
        # set_page_config can only be called once; ignore if already set
        pass

    load_css()
    render_header()


def render_header(active: Optional[str] = None) -> None:
    """Render a fixed, full-width header bar with logo or fallback brand text."""
    st.markdown('<div class="app-header"><div class="app-header-inner">', unsafe_allow_html=True)

    # Try to render a local logo; fall back to text brand if not found
    logo_rendered = False
    try:
        logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "Logo_SP.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=180)
            logo_rendered = True
    except Exception:
        logo_rendered = False

    if not logo_rendered:
        st.markdown('<div class="brand">Job Intelligence Platform</div>', unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

    # Spacer so body content doesn't hide under fixed header
    st.markdown('<div class="header-spacer"></div>', unsafe_allow_html=True)


