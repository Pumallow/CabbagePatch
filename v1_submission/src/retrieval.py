import os
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
import streamlit as st

# --------------------------- YOUTUBE API ------------------------------------
# ----- GAIN ACCESS -----
def get_youtube_client():
    YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY") or os.getenv("YOUTUBE_API_KEY")

    if not YOUTUBE_API_KEY:
        raise ValueError("Missing YOUTUBE_API_KEY")


    return build(
        "youtube",
        "v3",
        developerKey=YOUTUBE_API_KEY,
        cache_discovery=False
    )

# -------- PULL INFO -------------------------
def search_youtube(query, max_results=10):
    yt = get_youtube_client()
    search = yt.search().list(
        q=query,
        part="snippet",
        maxResults=max_results,
        type="video",
        safeSearch = "none",
        order = 'relevance',
        videoDuration = "any",
        regionCode = "US"
    ).execute()

    video_ids = [item["id"]["videoId"] for item in search["items"]]
    # print("VIDEO IDS:", video_ids)

    # print("SEARCH RESULTS COUNT:", len(search.get("items", [])))
    # for item in search.get("items", []):
    #     print("TITLE:", item["snippet"]["title"])

    return video_ids

def fetch_video_details(video_ids):
    yt = get_youtube_client()
    details = yt.videos().list(
        part="snippet,contentDetails,statistics",
        id=",".join(video_ids)
    ).execute()
    return details["items"]

def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([t["text"] for t in transcript])
        return text
    except:
        return ""
    

# ------------------ EXTRACT TIME --------------------------------

import isodate

def parse_duration_to_minutes(iso_duration: str):
    try:
        duration = isodate.parse_duration(iso_duration)
        return duration.total_seconds() / 60
    except:
        return 0


# ----------------------- YOUTUBE Vector Conversion ---------------------------------------------
def youtube_retrieval(query):
    video_ids = search_youtube(query)
    details = fetch_video_details(video_ids)

    docs = []
    for item in details:
        vid = item["id"]
        title = item["snippet"]["title"]
        desc = item["snippet"]["description"]
        transcript = fetch_transcript(vid)

        content = f"""
        Title: {title}
        Description: {desc}
        Transcript: {transcript[:800]}
        URL: https://www.youtube.com/watch?v={vid}
        """

        duration_iso = item["contentDetails"]["duration"]
        duration_minutes = parse_duration_to_minutes(duration_iso)

        docs.append(Document(
            page_content=content,
            metadata={
                "title": title,
                "description": desc,
                "url": f"https://www.youtube.com/watch?v={vid}",
                "duration": duration_minutes
            }
        ))

    return docs


