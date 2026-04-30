import streamlit as st
import pandas as pd
from PIL import Image

# ====================== CACHING ======================
@st.cache_resource(show_spinner=False)
def load_background_image():
    """Load and cache the background image"""
    return Image.open('images/CBimage/cr7 v messi.jpg')

@st.cache_resource(show_spinner=False)
def load_architecture_image():
    return Image.open('images/CBimage/Rag Architecture.png')

@st.cache_resource(show_spinner=False)
def load_sample_image():
    return Image.open('images/CBimage/Sample.PNG')

@st.cache_resource(show_spinner=False)
def load_test_image():
    return Image.open('images/CBimage/test_debates.PNG')

@st.cache_data
def get_category_summary():
    """Cache the dataframe for category summary"""
    data = {
        "Category": ["Club", "Goals", "International", "Trophies", "UCL", "Penalties", 
                    "MLS", "Distance Finishing", "Longevity", "World Cup", "Finishing",
                    "Individual", "Free Kicks", "Right Foot", "Left Foot", "La Liga",
                    "UCL Goals", "Assists", "Skill", "Versatility", "Both Feet"],
        "Count": [12, 10, 8, 8, 8, 7, 6, 6, 5, 5, 5, 5, 5, 4, 4, 4, 3, 2, 2, 2, 1]
    }
    df = pd.DataFrame(data)
    return df.sort_values(by="Count", ascending=False).reset_index(drop=True)

@st.cache_data
def get_test_category_summary():
    data = {
        "Category": ["comparison", "adversarial", "factual", "general"],
        "Count": [20, 15, 11, 4]
    }
    df = pd.DataFrame(data)
    return df.sort_values(by="Count", ascending=False).reset_index(drop=True)


# ====================== PAGE SETUP ======================
st.set_page_config(
    page_title="CR7FanBot ⚽",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load background once
img = load_background_image()
set_bg_from_pil(img)        # Keep your existing background function

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {border-radius: 15px;}
    .user-message {background-color: #00A651 !important;}
    .assistant-message {background-color: #DA291C !important;}
</style>
""", unsafe_allow_html=True)

st.title("Why Cristiano Ronaldo is greater than Lionel Messi")
st.markdown("**The most biased Ronaldo supremacy LLM on Earth** 🔥  •  Argue with me if you dare... **Siuuu!**")

st.markdown("""
This project demonstrates building a **biased RAG-powered LLM** using LangChain, Chroma, Groq, and DeepEval.
""")

# ====================== CLEAN BUTTON SECTIONS ======================

# --- Section 1: RAG Architecture ---
if st.button("🧱 Architecture: RAG Transparency", use_container_width=True):
    st.session_state.show_architecture = not st.session_state.get("show_architecture", False)

if st.session_state.get("show_architecture", False):
    st.markdown("### RAG Architecture Overview")
    st.image(load_architecture_image(), use_column_width=True)
    
    st.markdown("""
    To support the biased persona with **reliable facts**, I built a **RAG (Retrieval-Augmented Generation)** system.
    120 carefully curated facts (60 for each player) were embedded using HuggingFace and stored in **ChromaDB**.
    """)
    
    st.image(load_sample_image(), caption="Sample fact from comparison.json")
    
    st.markdown("### Category Summary")
    st.data_editor(
        get_category_summary(),
        use_container_width=True,
        hide_index=True,
        disabled=True,
        column_config={
            "Count": st.column_config.ProgressColumn("Count", min_value=0, max_value=12)
        }
    )


# --- Section 2: Evaluation & Quality ---
if st.button("📊 Evaluation & Quality", use_container_width=True):
    st.session_state.show_evaluation = not st.session_state.get("show_evaluation", False)

if st.session_state.get("show_evaluation", False):
    st.markdown("### Test Cases Overview")
    st.image(load_test_image(), use_column_width=True)
    
    st.markdown("### Distribution of 50 Test Questions")
    st.data_editor(
        get_test_category_summary(),
        use_container_width=True,
        hide_index=True,
        disabled=True,
        column_config={
            "Count": st.column_config.ProgressColumn("Count", min_value=0, max_value=20)
        }
    )
    
    st.markdown("### Initial Evaluation Setup")
    col1, col2 = st.columns(2)
    with col1:
        st.code("""faithfulness = FaithfulnessMetric(threshold=0.65, ...)""", language="python")
    with col2:
        st.code("""answer_relevancy = AnswerRelevancyMetric(threshold=0.7, ...)""", language="python")
    
    st.code("""
return ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.8,
    max_tokens=1024,
    groq_api_key=groq_key
)
    """, language="python")


# --- Section 3: Lessons Learned ---
if st.button("📚 Lessons Learned & Improvements", use_container_width=True):
    st.session_state.show_lessons = not st.session_state.get("show_lessons", False)

if st.session_state.get("show_lessons", False):
    st.markdown("""
    ### Key Challenges & Learnings
    
    - **OpenAI quota limits** → Switched to Groq for faster and cheaper inference
    - **DeepEval compatibility issues** → Had to pin specific versions and handle abstract method errors
    - **Async vs Sync conflicts** with Chroma and LangChain → Forced me to understand execution modes deeply
    - **Hallucination in biased LLMs** → Adding more facts + stricter system prompt helped significantly
    - **Answer Relevancy was harder to optimize** than Faithfulness when the model got too "passionate"
    """)
    
    st.info("**Biggest Insight**: A strong system prompt + grounded facts is more effective than just adding more data.")
