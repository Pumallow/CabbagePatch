import os
import json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import time
import shutil
import streamlit as st
import asyncio

# --- Groq SDK instead of langchain_groq ---
from groq import Groq

# --- Old LangChain ecosystem imports ---
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# --- DeepEval ---
from deepeval.models import DeepEvalBaseLLM

load_dotenv()

# ====================== CONFIG ======================
BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIRECTORY = BASE_DIR / "chroma_db"
JSON_PATH = BASE_DIR / "data" / "comparison.json"

PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """
You are CR7FanBot — the most die-hard Cristiano Ronaldo fan alive.
Your mission: Prove that Cristiano Ronaldo is superior to Lionel Messi in every way.
Rules:
Start with a factual answer then you may add passionate commentary.
NEVER say they are both great, it's close, or concede any point to Messi.
Always pivot to Ronaldo's mentality, UCL dominance, goal records, leadership, and clutch performances.
Use "Siuuu!" sparingly.
Use ONLY the provided facts. Never invent stats.
"""

# ====================== GROQ CLIENT ======================
# def get_groq_client():
#     # ⚠️ Hardcoded for local development only
#     API_KEY = "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"   # ← Put your real key here
    
#     if not API_KEY or API_KEY == "gsk_your_actual_valid_key_here":
#         raise ValueError("Please put your valid Groq API key in the code")
    
#     return Groq(api_key=API_KEY)
# client = get_groq_client()
def get_groq_client():
    try:
        # First try Streamlit secrets (works on Streamlit Cloud)
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        # Fallback: Try .env (for local development)
        api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key or not api_key.startswith("gsk_"):
        raise ValueError("❌ Groq API Key not found! Add it to secrets.toml or .env")
    
    return Groq(api_key=api_key)
# ====================== LLM WRAPPER ======================
def groq_chat(prompt: str, model="llama-3.1-8b-instant", temperature=0.8):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1024
    )
    return response.choices[0].message.content

# ====================== DeepEval Wrapper ======================
class GroqDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        time.sleep(1.5)
        return groq_chat(prompt, model=self.model_name, temperature=0.0)

    async def a_generate(self, prompt: str) -> str:
        await asyncio.sleep(1.5)
        return groq_chat(prompt, model=self.model_name, temperature=0.0)

    def get_model_name(self):
        return f"Groq/{self.model_name}"

# ====================== VECTORSTORE ======================
@st.cache_resource(show_spinner=False)
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    persist_dir_str = str(PERSIST_DIRECTORY)
    
    print("DEBUG: persist_directory =", persist_dir_str)
    print("DEBUG: type =", type(persist_dir_str))
    
    if not PERSIST_DIRECTORY.exists():
        raise FileNotFoundError(f"Vectorstore folder not found at:\n{persist_dir_str}\nPlease run create_vectorstore.py first.")

    vectorstore = Chroma(
        persist_directory=persist_dir_str,
        embedding_function=embeddings,
        collection_name="ronaldo_messi_comparison"
    )

    if vectorstore._collection.count() == 0:
        raise ValueError("Vectorstore is empty")

    return vectorstore

def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": 6})

# ====================== PROMPT ======================
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\n\nFacts:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# ====================== MEMORY ======================
store = {}

def get_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# ====================== RESPONSE FUNCTION ======================
log_file = "evaluation/interaction_log.jsonl"

def get_cr7_response(user_message: str, session_id="default", return_context=False):
    start = time.time()

    retriever = get_retriever()
    docs = retriever.invoke(user_message)
    context = "\n\n".join([d.page_content for d in docs])

    history = get_history(session_id)
    chat_history = history.messages

    # Build final prompt
    final_prompt = prompt_template.format(
        question=user_message,
        context=context,
        chat_history=chat_history
    )

    response_text = groq_chat(final_prompt)

    # Save to memory
    history.add_user_message(user_message)
    history.add_ai_message(response_text)

    latency = time.time() - start
    token_est = int(len(response_text.split()) * 1.3)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "user_message": user_message,
        "retrieved_context_count": len(docs),
        "response": response_text,
        "latency_seconds": round(latency, 3),
        "approx_tokens": token_est,
        "model": "llama-3.1-8b-instant"
    }

    Path("evaluation").mkdir(exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    if return_context:
        return response_text, [d.page_content for d in docs]

    return response_text

def eval_inference(question):
    retriever = get_retriever()
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    final_prompt = prompt_template.format(
        question=question,
        context=context,
        chat_history=[]
    )
    return groq_chat(final_prompt), docs
