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



def set_bg_from_pil(img):
    # 1. Convert PIL image to BytesIO
    buffered = BytesIO()
    img.save(buffered, format="PNG") # Use PNG or JPEG
    
    # 2. Encode to Base64 string
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # 3. Create the CSS
    # Use [data-testid="stAppViewContainer"] to target the main app container
    page_bg_img = f'''
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{img_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
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
