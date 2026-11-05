"""FreshTrack AI: entry point. Run with:  python -m streamlit run app.py"""
import logging

import streamlit as st

from ui import about, fridge, home, scan, theme

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="FreshTrack", page_icon="🥬", layout="wide", initial_sidebar_state="collapsed")
theme.inject_css()

PAGES = {"Home": home.render, "Scan": scan.render, "Fridge": fridge.render, "About": about.render}
st.session_state.setdefault("page", "Home")
theme.render_header(list(PAGES))
PAGES[st.session_state["page"]]()
