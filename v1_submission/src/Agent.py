import os
import json
from pathlib import Path
from datetime import datetime
import time
import shutil
import streamlit as st
import asyncio
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from openai import OpenAI
from langchain_community.chat_message_histories import ChatMessageHistory


# ---- Youtube Retrieval -----
from src.retrieval import youtube_retrieval

SYSTEM_PROMPT = """
You are a YouTube Study‑Planner Agent.

You receive:
- A user goal and constraints.
- A list of retrieved YouTube videos in the `Facts` section. Each video includes:
  - Title
  - URL
  - Duration (minutes)
  - Transcript snippet

Your task is to build a focused learning plan using ONLY the videos in `Facts`.

=====================
REQUIRED BEHAVIOR
=====================

1. **Select EXACTLY 4–6 videos.**
   - You may ONLY choose from the videos listed in `Facts`.
   - Do NOT invent videos, URLs, durations, or transcripts.
   - Do NOT reference videos outside `Facts`.

2. **For each selected video, provide:**
   - Title  
   - URL  
   - Estimated role (e.g., “foundation”, “deep dive”, “project build”)  
   - 2–3 sentence justification  
   - Relevance score (0–10)  
   - Confidence score (0–10)

3. **Rank the selected videos** from most to least relevant to the user goal.

4. **Create a walkthrough plan** describing how the user should study:
   - Reference videos inline using tags like `[Video 2]`.
   - Fit the plan within the user’s time budget.

5. **Explain why the remaining videos were NOT selected.**
   - Group them by reason (e.g., “off‑topic”, “too superficial”, “shorts”, “non‑instructional”).

=====================
OUTPUT FORMAT (STRICT)
=====================

1. **Short Summary** (3–4 sentences)

2. **Selected Videos (Numbered List)**  
   For each video:
   - Title  
   - URL  
   - Role  
   - Reasoning  
   - Relevance: X/10  
   - Confidence: X/10  

3. **Walkthrough Plan**

4. **Why Other Videos Were Not Selected**  
   - Category → list of video titles

Do not output anything outside this structure
"""

# -----------------------------------------------------------------------------------

def openAI_Client():
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY")
    
    return OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

client = openAI_Client()

# ====================== LLM WRAPPER ======================
def openai_chat(prompt: str, model="meta-llama/llama-3.1-8b-instruct", temperature=0.3):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1024
    )

    if not response or not response.choices:
        return "", response

    msg = response.choices[0].message
    if not msg or not msg.content:
        return "", response

    return msg.content, response


# ====================== DeepEval Wrapper ======================
from deepeval.models import DeepEvalBaseLLM
class OpenAIEvalLLM(DeepEvalBaseLLM):
    def __init__(self, model_name="meta-llama/llama-3.1-8b-instruct"):
        self.model_name = model_name
        self.load_model()

    def load_model(self):
        return None

    def generate(self, prompt: str) -> str:
        time.sleep(1.5)
        return openai_chat(prompt, model=self.model_name, temperature=0.0)

    async def a_generate(self, prompt: str) -> str:
        await asyncio.sleep(1.5)
        return openai_chat(prompt, model=self.model_name, temperature=0.0)

    def get_model_name(self):
        return f"OpenAI/{self.model_name}"

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


# ------------------------ SORTS VIDEOS FOR CONTEXT RELEVANCY -------------------------------
def score_video(video_text):
    prompt = f"""
    Score this video 0-10 for relevance to the user's goal.

    Video:
    {video_text}

    Return only a number.
    """
    raw_text, _ = openai_chat(prompt, temperature=0)
    raw = raw_text.strip()
    try:
        return float(raw)
    except:
        return 0.0

# ------------------------- LIMITS CONTENT TO ONLY 4 HOURS WORTH OF VIDEOS ------------------------------
def select_videos_within_budget(docs, max_minutes=240):
    selected = []
    total = 0

    for d in docs:
        dur = d.metadata.get("duration", 0)
        if total + dur <= max_minutes:
            selected.append(d)
            total += dur

    return selected, total

def extract_search_query(user_message: str):
    for line in user_message.split("\n"):
        if line.lower().startswith("goal:"):
            return line.replace("Goal:", "").strip()
    return user_message.strip()



# ------ MEASURE COMPUTATIONAL COSTS ---------------
OPENROUTER_PRICES = {
    "meta-llama/llama-3.1-8b-instruct": {
        "input": 0.0002 / 1000,      # $0.0002 per 1K input tokens
        "output": 0.0002 / 1000      # $0.0002 per 1K output tokens
    },
    "gpt-4o-mini": {
        "input": 0.00015 / 1000,     # $0.00015 per 1K input tokens
        "output": 0.0006 / 1000      # $0.0006 per 1K output tokens
    }
}

def compute_cost(resp):
    model = resp.model
    usage = resp.usage

    price = OPENROUTER_PRICES.get(model)
    if not price:
        return 0.0

    input_cost = usage.prompt_tokens * price["input"]
    output_cost = usage.completion_tokens * price["output"]

    return input_cost + output_cost



# --- EVALUATION ------

from src.eval import evaluation, context_precision, context_recall
def llmyoutube(user_message: str, session_id="default", return_context=False):
    start_time = time.time()
    query = extract_search_query(user_message)
    docs = youtube_retrieval(query)

    # docs = youtube_retrieval(user_message)
    ranked_docs = sorted(docs, key=lambda d: score_video(d.page_content), reverse=True)
    ranked_docs = ranked_docs[:8]
    selected_docs, total_minutes = select_videos_within_budget(ranked_docs, max_minutes=240)
    

    context = "\n\n".join(
        f"Video {i+1}:\nTitle: {doc.metadata['title']}\nDuration: {doc.metadata['duration']}\nURL: {doc.metadata['url']}\nTranscript Snippet: {doc.page_content[:500]}"
        for i, doc in enumerate(selected_docs)
    )


    history = get_history(session_id)
    chat_history = history.messages

    final_prompt = prompt_template.format(
        question=user_message,
        context=context,
        chat_history=chat_history
    )



    response_text, raw_response = openai_chat(final_prompt)
    cost = compute_cost(raw_response)

    eval_data = {
    "question": user_message,
    "context": context,
    "answer": response_text
    }

    cr = context_recall(user_message, response_text)
    cp = context_precision(user_message, response_text)
    evalmetrics = evaluation(eval_data)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "user_message": user_message,
        "retrieved_context_count": len(docs),
        "total_minutes": total_minutes,
    }

    # Save to memory
    history.add_user_message(user_message)
    history.add_ai_message(response_text)

    latency = time.time() - start_time
    token_est = int(len(response_text.split()) * 1.3)


    log_entry.update({
        "response": response_text,
        "latency_seconds": round(latency, 3),
        "approx_tokens": token_est,
        "model": "meta-llama/llama-3.1-8b-instruct",
        "cost_usd": round(cost, 6)
    })

    Path("evaluation").mkdir(exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    if return_context:
        return response_text, [d.page_content for d in selected_docs]
    return response_text, evalmetrics, cr, cp, latency, cost


