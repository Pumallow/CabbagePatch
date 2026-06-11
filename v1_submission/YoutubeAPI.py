import json
from src.Agent import llmyoutube, extract_search_query
from src.retrieval import search_youtube, youtube_retrieval
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

TEST_FILE = BASE_DIR / "test_set" / "test_inputs.json"

with open(TEST_FILE, "r") as f:
    data = json.load(f)

personas = data[:3]   # first item in the list

def build_prompt(persona):
    return f"""
Goal: {persona['goal']}
Time Budget: {persona['time_budget_minutes']} minutes
Background: {persona['user_context']['background']}
Known: {', '.join(persona['user_context']['known'])}
Unknown: {', '.join(persona['user_context'].get('unknown', []))}
Constraints: {persona['user_context']['constraints']}
"""

results = []
for idx, persona in enumerate(personas, start=1):
    print(f"\n\n==============================")
    print(f"=== RUNNING PERSONA {idx} ===")
    print(f"==============================\n")

    prompt = build_prompt(persona)
    print("PERSONA PROMPT:")
    print(prompt)

    # Run full agent
    print("\nAGENT OUTPUT:\n")
    response, metrics, cr, cp, latency, cost = llmyoutube(prompt, session_id=f"persona_{idx}")
    print(response)

    results.append([metrics, cr, cp, cost])
    

print(results)



# from googleapiclient.discovery import build
# import streamlit as st

# def test_youtube_connection():
#     try:
#         key = st.secrets["YOUTUBE_API_KEY"]
#     except:
#         print("Key not found in Streamlit secrets")
#         return

#     yt = build("youtube", "v3", developerKey=key)

#     # Simple test: search for "react tutorial"
#     req = yt.search().list(
#         q="react tutorial",
#         part="snippet",
#         maxResults=1
#     )
#     res = req.execute()

#     print("API Response:", res)
#     print("Items returned:", len(res.get("items", [])))
# test_youtube_connection()



