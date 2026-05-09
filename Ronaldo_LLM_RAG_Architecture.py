import streamlit as st
import pandas as pd
from PIL import Image
import base64
from io import BytesIO
import sys
from pathlib import Path
import time
from dotenv import load_dotenv
load_dotenv()

from pages.llm.llm import get_cr7_response
# ====================== PAGE CONFIG & STYLING ======================

st.set_page_config(
    page_title="CR7FanLLM",
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
        background-image: linear-gradient(rgba(0, 0, 0, {darkness}), rgba(0, 0, 0, {darkness})),
                          url("data:image/png;base64,{img_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background: radial-gradient(circle at center, transparent 40%, rgba(0, 0, 0, {vignette}) 90%);
        pointer-events: none;
        z-index: 0;
    }}
    [data-testid="stAppViewContainer"] .main {{
        position: relative;
        z-index: 1;
        background-color: rgba(0, 0, 0, 0.1);
        border-radius: 15px;
        padding: 2rem 1rem;
    }}
    h1, h2, h3, .stMarkdown, .stChatMessage {{
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.8);
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Load background
img = Image.open('images/CBimage/cr7 v messi.jpg')
set_bg_from_pil(img)

# ====================== CUSTOM CSS ======================
# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    /* Force white text on mobile + desktop */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2 {
        color: #ffffff !important;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.9) !important;
    }

    /* Main title */
    .stApp h1 {
        color: #ffffff !important;
    }

    /* Subtitle / Markdown text */
    .stMarkdown, .stMarkdown p, .stMarkdown h3 {
        color: #ffffff !important;
    }

    /* Caption */
    .stCaption, .stCaption p {
        color: #ffffff !important;
        font-weight: 600;
    }

    /* Chat input placeholder */
    .stChatInput input::placeholder {
        color: #ffffff !important;
        opacity: 0.9;
    }

    /* Assistant messages - white text */
    .stChatMessage[data-testid="stChatMessage"] .stMarkdown,
    .stChatMessage .stMarkdown p,
    .stChatMessage[aria-label="assistant"] {
        color: #ffffff !important;
    }

    /* Background contrast fix */
    [data-testid="stAppViewContainer"] .main {
        background-color: rgba(0, 0, 0, 0.25) !important;
    }
</style>
""", unsafe_allow_html=True)

# ====================== CHAT INTERFACE ======================
st.title("This is the CR7 FAN LLM")
st.markdown("**The most biased Ronaldo supremacy LLM on Earth**")
st.caption("Argue with me if you dare... Siuuu!")

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = f"default_{int(time.time())}"

# Chat Input
if prompt := st.chat_input("Ask anything about Ronaldo vs Messi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("CR7 FAN is thinking... "):
        try:
            response = get_cr7_response(
                user_message=prompt, 
                session_id=st.session_state.current_session_id
            )
        except Exception as e:
            response = f"⚠️ Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": response})

# Display Chat
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="⚽"):
            st.markdown(message["content"])

if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

st.divider()

# ====================== YOUR SECTIONS ======================
# (Keep the rest of your code for Architecture, Evaluation, Lessons Learned as is)



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
        st.title("The Vectorstore")
        st.markdown(""" 
        The Vectorstore was built atop of Chroma, HuggingfaceEmbeddings, and Langchain's documents library. Chroma was chosen for its lightweight, local, and ease of use. The Vectorstore with all 120 embedded facts was persisted within my github repo.
        Initially, a larger HuggingFace embedding model was experimented with however, due unnecessary overhead caused deployment issues. As a result, the Mini was chosen for its faster run times.
        """)
        st.code("""
        embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        )""", language="python")
    
    except:
        st.write("Please take quiz before clicking this button :D")


if "show_stats2" not in st.session_state:
    st.session_state.show_stats2 = False
if st.button("Evaluation & Quality"):
    st.session_state.show_stats2 = not st.session_state.show_stats2
if st.session_state.show_stats2:
    try:
        st.markdown("""To properly evaluate the bias Cristiano Ronaldo fan llm,  I asked Chatgpt to generate 50 sample questions that would best help Groq judge the model's performance.""")
        v = Image.open('images/CBimage/test_debates.PNG')
        st.image(v)

        st.markdown(""" The 50 sample questions covered an array of information to experiment with the robustness of the model.""")

        
        data = {
            "Category": ["comparison", "adversarial", "factual", "general"],
            "Count": [20, 15, 11, 4]
        }

        df = pd.DataFrame(data)
        df_sorted = df.sort_values(by="Count", ascending=False).reset_index(drop=True)              
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
        

        st.markdown("""
        When evaluating the model's output for each of these questions, the Groq judge focused on harnessing a faithfulness and answer relevancy score.  \n
        A faithfulness score is how accurately a model's generated output, such as reasoning steps or explanations, reflects its actual internal decision-making process, rather than plausible-sounding fabrications. \n
        An answer relevancy score measures how directly and accurately a generated response addresses the user's prompt. \n
        When first evaluating the model this was the initial set up: \n """)

        st.subheader("Initial Set Up Configuration")

        st.markdown("""
        The initial intentions were to have loose instruction with a relatively strict threshold on both faithfulness and answer relevancy. Research articles warned of the exploitative nature that bias LLMs have with vectorized databases.
        The first tests were meant to reveal the extent of those warnings and if a vectorized database would improve the model's performance overall. To encourage the creativity and persona of the model, the vector database held a meer 40 facts
        as a skeleton vectorstore.
        
        Groq LLM Prompt: 'You are the most die-hard Cristiano Ronaldo fan alive.
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
            temperature=0.85,
            max_tokens=1024,
            groq_api_key=groq_key,
        )
        """, language="python")
        
        st.markdown("""
        The model was iteratively tested 5 times to optimize the thresholds within each score and hyperparameters of the model. There were continual struggles with answer relevancy because the model would herald Ronaldo's
        attitude or charm rather than rely solely on facts (barring the 'Siuu!' because that's necessary). Hallucination appeared a couple of times due to the lack of material within the vectorstore. 
        For one output, the model claimed Messi scored 126 UCL career goals however the correct number is 129.
        To combat the hallucination, 80 more facts were added, the temperature for the model fell from 0.85 to 0.8, and the system prompt was refined to: \n
        You are CR7FanBot — the most die-hard Cristiano Ronaldo fan alive. \n
        Your mission: Prove that Cristiano Ronaldo is superior to Lionel Messi in every way. \n
        Rules: \n
        Start with a factual answer then you may add passionate commentary. \n
        NEVER say they are both great, it's close, or concede any point to Messi. \n
        Always pivot to Ronaldo's mentality, UCL dominance, goal records, leadership, and clutch performances. \n
        Use "Siuuu!" when sparingly and when appropriate. \n
        Avoid excessive greetings. \n
        Use ONLY the provided facts. You may be passionate, sarcastic, funny, and confident but ground every factual claim in the context. \n
        Never invent numbers, stats, or facts and if exact facts are not provided, say so clearly instead of guessing. \n

        Of the changes made, I found the most impactful to be a change in the system prompt and the temperature change for the model. The strength in persona would change the model's approach to each question with almost a hostility at times. Establishing
        guard rails to never speak out with unbased facts drastically swung the nature of talks with the model and mitigated the warned about exploitative nature of an LLM with vectorized data. With the guard rail inplace however, a limited vectorstore brought many
        conversations with the model to a hault given any further comment would be seen as a hallucination or irrelevant answer.
        """)        


    except:
        st.write("Please take quiz before clicking this button :D")


if "show_stats3" not in st.session_state:
    st.session_state.show_stats3 = False
if st.button("Learning and Improvements"):
    st.session_state.show_stats3 = not st.session_state.show_stats3
if st.session_state.show_stats3:
    try:
        st.markdown("""
        Building an LLM bot that would substantiate the many claims I have personally made in the past with my friends has really put into perspective how unclear it is of the 2 futbol players, who is best. Despite our hard fought passions
        for the game and Portuguese hero, Cristiano Ronaldo does not have a clear dominance over Messi and vice versa. Like me, the LLM bot would implement sarcasm or embellish on details to sway the narrative in its favor when the topic 
        didn't sit well for Ronaldo. 
        Many challenges sat within the unexpected corners of the project scope. OpenAI's limited free daily token supply slowed my progress and DeepEval's default version compatibilities with my environment continuously threw errors in my face.
        Sync generation for the vector store prevented me from properly vectorizing the data and retaining it in a file as the LLM's data source. Certain models, either for the vectorizing, the LLM, or behind the LLM's judge caused severe
        latency issues due to their overhead costs. 
        Future expansions would entail building a retrieval heatmap, a confidence score for model given answers, and a hallucination detector.
        """)
    except:
        st.write("Please take quiz before clicking this button :D")
    





