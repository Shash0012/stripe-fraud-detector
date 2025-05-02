# ui_layout.py

import streamlit as st

def render_sidebar():
    st.sidebar.title("🔍 Navigation")
    st.sidebar.markdown("Upload data or generate mock samples to test fraud detection.")
    st.sidebar.markdown("---")

def render_header():
    st.title("💳 Stripe Fraud Detection App")
    st.markdown("This app helps identify potential credit card fraud using a trained ML model.")
    st.markdown("---")

def render_footer():
    st.markdown("---")
    st.caption("Made with ❤️ for your final project. Powered by Streamlit.")
