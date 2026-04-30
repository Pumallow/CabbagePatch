import streamlit as st
import pandas as pd
import json
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO

# Page config & styling
st.set_page_config(
    page_title="CR7FanBot ⚽",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed"
)



def set_bg_from_pil(img, darkness=0.65, vignette=0.4):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    page_bg_img = f'''
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: 
            linear-gradient(rgba(0, 0, 0, {darkness}), rgba(0, 0, 0, {darkness})), 
            url("data:image/png;base64,{img_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Optional: subtle vignette (darker corners) for more "silhouette" feel */
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(
            circle at center,
            transparent 40%,
            rgba(0, 0, 0, {vignette}) 90%
        );
        pointer-events: none;
        z-index: 0;
    }}

    /* Make sure main content sits above the overlay */
    [data-testid="stAppViewContainer"] .main {{
        position: relative;
        z-index: 1;
        background-color: rgba(0, 0, 0, 0.1);   /* very light extra dark layer if needed */
        border-radius: 15px;
        padding: 2rem 1rem;
    }}

    /* Improve text readability */
    h1, h2, h3, .stMarkdown, .stChatMessage {{
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.8);
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
# Example Usage:
# Open your image using PIL
img = Image.open('images/CBimage/cr7 v messi.jpg')
set_bg_from_pil(img)



# Custom CSS for football vibe
st.markdown("""
<style>
    .stChatMessage {border-radius: 15px;}
    .user-message {background-color: #00A651 !important;}   /* Green like Portugal */
    .assistant-message {background-color: #DA291C !important;} /* Red like Manchester United */
</style>
""", unsafe_allow_html=True)

st.title("Who is the best futbol player? Cristiano Ronaldo or Lionel Messi")
st.markdown("**The most biased Ronaldo supremacy LLM on Earth** 🔥\n\nArgue with me if you dare... Siuuu!")
