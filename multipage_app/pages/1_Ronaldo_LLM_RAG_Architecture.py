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

st.title("Why Cristiano Ronaldo is greater than Lionel Messi. An LLM RAG Architecture story.\n")

st.markdown("""
Most every futbol fan has argued this one debate at one point or another but if you are not particularly a futbol fan this can set the stage. Cristiano Ronaldo and Lionel Messi
stand as to modern day futbolling giants that have raised the bar for anyone pushing to be the "best" in the sport. Having won many cups, accollades, and both scoring over 900 goals, these 2 individuals are continually juxtaposed
as the better talent despite their difference in playstyle and positions.
\n
To better understand the building, testing, and deployment of an LLM atop of a RAG Architecture design, I combined my love for futbol with my love for data.
""")

st.markdown("**The most biased Ronaldo supremacy LLM on Earth** 🔥\n\nArgue with me if you dare... Siuuu!")
if "show_stats" not in st.session_state:
    st.session_state.show_stats = False

# Button that toggles on every click
if st.button("Architecture: RAG Transparency"):
    st.session_state.show_stats = not st.session_state.show_stats

if st.button("Evaluation & Quality"):
    st.session_state.show_stats = not st.session_state.show_stats

if st.button("Learning and Improvements"):
    st.session_state.show_stats = not st.session_state.show_stats
    
if st.session_state.show_stats:
    try:
        s = Image.open('images/CBimage/Rag Architecture.png')
        st.image(s)
    except:
        st.write("Please take quiz before clicking this button :D")

if st.session_state.show_stats:
    try:
        st.markdown("""
        Most every futbol fan has argued this one debate at one point or another but if you are not particularly a futbol fan this can set the stage. Cristiano Ronaldo and Lionel Messi
        stand as to modern day futbolling giants that have raised the bar for anyone pushing to be the "best" in the sport. Having won many cups, accollades, and both scoring over 900 goals, these 2 individuals are continually juxtaposed
        as the better talent despite their difference in playstyle and positions.
        \n
        To better understand the building, testing, and deployment of an LLM atop of a RAG Architecture design, I combined my love for futbol with my love for data.
        """)
    except:
        st.write("Please take quiz before clicking this button :D")

if st.session_state.show_stats:
    try:
        st.markdown("""
        Most every futbol fan has argued this one debate at one point or another but if you are not particularly a futbol fan this can set the stage. Cristiano Ronaldo and Lionel Messi
        stand as to modern day futbolling giants that have raised the bar for anyone pushing to be the "best" in the sport. Having won many cups, accollades, and both scoring over 900 goals, these 2 individuals are continually juxtaposed
        as the better talent despite their difference in playstyle and positions.
        \n
        To better understand the building, testing, and deployment of an LLM atop of a RAG Architecture design, I combined my love for futbol with my love for data.
        """)
    except:
        st.write("Please take quiz before clicking this button :D")


