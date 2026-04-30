import os
import json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import time
import shutil
import streamlit as st
import asyncio
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from deepeval.models import DeepEvalBaseLLM

load_dotenv()
@st.cache_resource(show_spinner=False)
# ====================== CONFIG ======================
PERSIST_DIRECTORY = "./chroma_db"
JSON_PATH = "data/comparison.json"

# Strong CR7 Persona Prompt
SYSTEM_PROMPT = """
You are CR7FanBot — the most die-hard Cristiano Ronaldo fan alive.
Your mission: Prove that Cristiano Ronaldo is superior to Lionel Messi in every way.
Rules:
Start with a factual answer then you may add passionate commentary.
NEVER say they are both great, it's close, or concede any point to Messi.
Always pivot to Ronaldo's mentality, UCL dominance, goal records, leadership, and clutch performances.
Use "Siuuu!" when sparingly and when appropriate.
Avoid excessive greetings.
Use ONLY the provided facts. Do not invent new statistics. You may be passionate, sarcastic, funny, and confident but ground every factual claim in the context.
Never invent numbers, stats, or facts and if exact facts are not provided, say so clearly instead of guessing.
"""

class GroqDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.model_name = model_name
        self._llm = None

    def load_model(self):
        if self._llm is None:
            self._llm = ChatGroq(
                model=self.model_name,
                temperature=0.0,          # Must be 0 for evaluation
                max_tokens=800,
                groq_api_key=os.getenv("GROQ_API_KEY"),
            )
        return self._llm

    def generate(self, prompt: str) -> str:
        time.sleep(1.5)
        response = self.load_model().invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        await asyncio.sleep(1.5)
        response = await self.load_model().ainvoke(prompt)
        return response.content

    def get_model_name(self) -> str:
        return f"Groq/{self.model_name}"



# ====================== LLM ======================
def get_llm():
    groq_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY not found in secrets or .env")
    
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.8,
        max_tokens=1024,
        groq_api_key=groq_key,
    )

llm = get_llm()

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not Path(PERSIST_DIRECTORY).exists():
        raise FileNotFoundError(f"Vectorstore not found at {PERSIST_DIRECTORY}. Run create_vectorstore.py first.")

    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name="ronaldo_messi_comparison"   # Good to specify collection name
    )
    
    count = vectorstore._collection.count()
    print(f"✅ Loaded vectorstore with {count} documents")
    
    if count == 0:
        raise ValueError("Vectorstore is empty!")
    
    return vectorstore

def get_retriever():
    """Get retriever - lazy initialization"""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": 6})


# ====================== PROMPT ======================
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\n\nUse the following facts to strengthen your arguments:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# ====================== CHAIN & MEMORY ======================
chain = prompt | llm

store = {}  # session history store

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# ====================== RESPONSE FUNCTION ======================
log_file = "evaluation/interaction_log.jsonl"

def get_cr7_response(user_message: str, session_id: str = "default", return_context: bool = False):
    start_time = time.time()
    
    # Safe RAG retrieval
    retriever = get_retriever()
    retrieved_docs = retriever.invoke(user_message)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    response = conversational_rag_chain.invoke(
        {"question": user_message, "context": context},
        config={"configurable": {"session_id": session_id}}
    )
    
    latency = time.time() - start_time
    
    # Rough token estimate (you can improve this later with exact Groq usage)
    token_count = len(response.content.split()) * 1.3  
    
    # Logging
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "user_message": user_message,
        "retrieved_context_count": len(retrieved_docs),
        "response": response.content,
        "latency_seconds": round(latency, 3),
        "approx_tokens": round(token_count),
        "model": "llama-3.1-8b-instant"
    }
    
    Path("evaluation").mkdir(exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    print(f"✅ Response in {latency:.2f}s | ~{round(token_count)} tokens")
    
    if return_context:
        return response.content, [doc.page_content for doc in retrieved_docs]
    return response.content


def eval_inference(question):
    retriever = get_retriever()
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    return chain.invoke({"question": question, "context": context, "chat_history": [] }).content, docs
