import streamlit as st
from PIL import Image

# ====================== CONFIG ======================
# st.set_page_config(
#     page_title="RL Projects | Marshal Turner",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# Hide default menu & footer
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ====================== CACHED IMAGE LOADING ======================
@st.cache_resource(show_spinner=False)
def load_img(path: str):
    return Image.open(path)

# Load once at startup (no repeated opens)
car_img      = load_img("images/ReinforcementLearning/car.png")
ppo_img      = load_img("images/ReinforcementLearning/PPO.png")
tracks_img   = load_img("images/ReinforcementLearning/racetracks.png")
map1         = load_img("images/ReinforcementLearning/Map 1.png")
map2         = load_img("images/ReinforcementLearning/Map 2.png")
map3         = load_img("images/ReinforcementLearning/Map 3.png")

# ====================== PDF DOWNLOADS ======================
@st.cache_data(show_spinner=False)
def load_pdf(filename: str):
    with open(f"images/ReinforcementLearning/{filename}", "rb") as f:
        return f.read()

pdf4 = load_pdf("Project4Paper.pdf")
pdf3 = load_pdf("Project3Paper.pdf")

# ====================== SIDEBAR ======================
st.sidebar.title("🔬 Reinforcement Learning")
project = st.sidebar.radio(
    "Choose Project",
    ["Project 4: DeepRacer PPO", "Project 3: Overcooked QMIX"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Marshal Turner • Georgia Tech OMSA")

# ====================== MAIN APP ======================
st.title("Reinforcement Learning Projects")
st.markdown("**Highlight reel of my RL work**")

# Download buttons (always visible)
col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button(
        label="📄 Download DeepRacer Paper (PDF)",
        data=pdf4,
        file_name="Project4_DeepRacer_PPO.pdf",
        mime="application/pdf"
    )
with col_dl2:
    st.download_button(
        label="📄 Download Overcooked QMIX Paper (PDF)",
        data=pdf3,
        file_name="Project3_Overcooked_QMIX.pdf",
        mime="application/pdf"
    )

st.markdown("---")

# ====================== PROJECT TABS ======================
if project == "Project 4: DeepRacer PPO":
    st.header("Project 4: DeepRacer – PPO Agent Prepping for the F1")
    
    st.markdown("""
    Trained a **PPO** agent for AWS DeepRacer across 3 tracks (reInvent2019-wide, reInvent2019, Vegas)  
    in Time-Trial, Object-Avoidance, and Head-to-Head modes (~80,000 episodes).
    """)
    
    intf = open(r"images/ReinforcementLearning/LapTrial.mp4", 'rb')
    intb = intf.read() 
    st.video(data=intb)
    
    # Hero images
    c1, c2, c3 = st.columns(3)
    with c1: st.image(car_img, caption="AWS DeepRacer Car", use_column_width=True)
    with c2: st.image(ppo_img, caption="PPO Architecture", use_column_width=True)
    with c3: st.image(tracks_img, caption="Race Tracks", use_column_width=True)

    st.markdown("""
    **Key Breakthrough**: “No Mercy, No Exploits” reward function  
    - Capped all bonuses  
    - Eliminated multiplicative cascades  
    - Killed 6 crawling/zigzag/wall-hugging exploits  

    **Results**:
    - Time-Trial: 76.3% max progress + clean laps (sub-16s potential)
    - Object-Avoidance: 26.3% max progress
    - Head-to-Head: Strong early race positioning
    """)

elif project == "Project 3: Overcooked QMIX":
    st.header("Project 3: Collaborative Onion Soup Delivery via QMIX")
    
    st.markdown("""
    Trained **two cooperative agents** using **QMIX** (monotonic value factorization + per-agent GRUs) 
    to deliver ≥7 onion soups across three Overcooked layouts.
    """)

    # Kitchen maps in nice columns
    c1, c2, c3 = st.columns(3)
    with c1: st.image(map1, caption="Cramped Room", use_column_width=True)
    with c2: st.image(map2, caption="Coordination Ring", use_column_width=True)
    with c3: st.image(map3, caption="Counter Circuit", use_column_width=True)

    st.markdown("""
    **Results** (150-episode evaluation):
    - Cramped Room: **3.2** soups
    - Coordination Ring: **0.8** soups
    - Counter Circuit: **0.0** soups

    **Lessons Learned**:
    - Dense reward shaping is extremely fragile in MARL
    - Over-penalization → policy collapse ("paralysis by analysis")
    - QMIX excels at role specialization but struggles with tight coordination
    """)

