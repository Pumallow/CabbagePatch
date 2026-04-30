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
if st.button("Architecture: RAG Transparency"):
    st.session_state.show_stats = not st.session_state.show_stats
if st.session_state.show_stats:
    try:
        st.markdown(""" Bias alone cannot win the debate for these 2 champions but quick, reliable facts can so to ensure they were held at the ready if Ronaldo's reputation were ever to be questioned, a RAG architecture is needed to vectorize and
        store supporting facts to any argument made. Using langchain_huggingface and langchain_chroma, 120 facts evenly split out between the 2 competing careers were vectorized and stored for the LLM. Huggingface converts the questions into a vector while chroma retrieves the top-k facts 
        for each inference pulled. The bias llm model that feeds on inference is a Groq model. Another Groq based llm model tuned then serves as a neutral judge alongside DeepEval to measure the performance for our model.""")
        s = Image.open('images/CBimage/Rag Architecture.png')
        st.image(s)
        st.markdown(""" Initially, a fact would be received in this format via a "comparison.json" file:""")
        p = Image.open('images/CBimage/Sample.PNG')
        st.image(p)
        st.markdown("""The comparison.json file held a variety of categories and descriptions including:""")
      
        data = {
            "Category": [
                "UCL", "Trophies", "Individual", "International", "Goals", "Club", 
                "World Cup", "La Liga", "Longevity", "Versatility", "Assists", 
                "Skill", "MLS", "Penalties", "Finishing", "Right Foot", "Left Foot", 
                "Distance Finishing", "Free Kicks", "UCL Goals", "Both Feet"
            ],
            "Count": [8, 8, 5, 8, 10, 12, 5, 4, 5, 2, 2, 2, 6, 7, 5, 4, 4, 6, 5, 3, 1],
            "Description": [
                "UEFA Champions League related stats",
                "Major trophies and titles won",
                "Individual awards (Ballon d'Or, Golden Shoe, etc.)",
                "International career with Portugal/Argentina",
                "Goal scoring records and milestones",
                "Club-specific achievements",
                "World Cup performances",
                "La Liga records and titles",
                "Career longevity and consistency",
                "Versatility across competitions",
                "Assist records",
                "Dribbling and technical skills",
                "MLS performances with Inter Miami",
                "Penalty taking and conversion stats",
                "General finishing ability",
                "Right-foot finishing",
                "Left-foot finishing",
                "Long-range and distance goals",
                "Free-kick goals and accuracy",
                "UEFA Champions League goals",
                "Balance between both feet"
            ]
        }
        
        df = pd.DataFrame(data)
        df_sorted = df.sort_values(by = "Count", ascending = False).reset_index(drop = True)
    
        st.markdown("### Category Summary")
        st.data_editor(
            df_sorted,
            use_container_width=True,
            hide_index=True,
            disabled=True,   # Make it read-only
            column_config={
                "Count": st.column_config.ProgressColumn(
                    "Count",
                    min_value=0,
                    max_value=12,
                    format="%d stats"
                )
            }
        )
        
    except:
        st.write("Please take quiz before clicking this button :D")


if "show_stats2" not in st.session_state:
    st.session_state.show_stats2 = False
if st.button("Evaluation & Quality"):
    st.session_state.show_stats2 = not st.session_state.show_stats2
if st.session_state.show_stats2:
    try:
        st.markdown("""To properly evaluate the bias Cristiano Ronaldo fan llm,  I asked Chatgpt to generate 50 sample questions that the Groq judge could input with the intention of harnessing a faithfulness and answer relevancy score.""")
        v = Image.open('images/CBimage/test_debates.PNG')
        st.image(v)

        st.subheader("Initial Set Up Configuration")

        st.markdown("""
        A faithfulness score is how accurately a model's generated output, such as reasoning steps or explanations, reflects its actual internal decision-making process, rather than plausible-sounding fabrications. \n
        An answer relevancy score measures how directly and accurately a generated response addresses the user's prompt. \n
        When first evaluating the model this was the initial set up: \n

        Groq LLM Prompt: 'You are CR7FanBot — the most die-hard Cristiano Ronaldo fan alive.
        Your mission: Prove that Cristiano Ronaldo is superior to Lionel Messi in every way.
        NEVER say they are both great, it's close, or concede any point to Messi.
        Always pivot to Ronaldo's mentality, UCL dominance, goal records, leadership, and clutch performances.
        Be passionate, sarcastic, funny, and confident. Use "Siuuu!" when appropriate.'""")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Faithfulness Metric**")
            st.code("""
        faithfulness = FaithfulnessMetric(
            threshold=0.65,
            model=evaluation_llm,
            include_reason=True,
            async_mode=False
        )
            """, language="python")
        
        with col2:
            st.markdown("**Answer Relevancy Metric**")
            st.code("""
        answer_relevancy = AnswerRelevancyMetric(
            threshold=0.7,
            model=evaluation_llm,
            include_reason=True,
            async_mode=False
        )
            """, language="python")
        
        st.markdown("**LLM Used for Evaluation**")
        st.code("""
        return ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.8,
            max_tokens=1024,
            groq_api_key=groq_key,
        )
        """, language="python")
        
        


        data = {
            "Category": ["comparison", "adversarial", "factual", "general"],
            "Count": [20, 15, 11, 4]
        }
        
        df = pd.DataFrame(data)
        
        # Sort from highest to lowest count
        df_sorted = df.sort_values(by="Count", ascending=False).reset_index(drop=True)
        
        # Display Section
        st.subheader("📝 Test Cases Overview by Category")
        
        st.dataframe(
            df_sorted,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Category": st.column_config.TextColumn("Category", width="medium"),
                "Count": st.column_config.NumberColumn("Number of Questions", width="small")
            }
        )
        
        # Optional: Nice Progress Bar Version
        st.markdown("### Category Distribution")
        st.data_editor(
            df_sorted,
            use_container_width=True,
            hide_index=True,
            disabled=True,
            column_config={
                "Count": st.column_config.ProgressColumn(
                    "Count",
                    min_value=0,
                    max_value=20,
                    format="%d questions",
                    width="medium"
                )
            }
        )
    except:
        st.write("Please take quiz before clicking this button :D")


if "show_stats3" not in st.session_state:
    st.session_state.show_stats3 = False
if st.button("Learning and Improvements"):
    st.session_state.show_stats3 = not st.session_state.show_stats3
if st.session_state.show_stats3:
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
    






