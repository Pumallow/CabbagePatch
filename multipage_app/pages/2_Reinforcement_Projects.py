import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


st.set_page_config(layout="wide", page_title="Reinforcement Learning")
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.markdown("""
<div style = 'text-align: center; font-size: 30px'>PPO Autonomous Driving Project""", unsafe_allow_html=True)



with open("images/ReinforcementLearning/ProjectPaper3.pdf", "rb") as pdf_file:
    PDFByte = pdf_file.read()

with open("images/ReinforcementLearning/ProjectPaper4.pdf", "rb") as pdf_file:
    PDFByte = pdf_file.read()


st.markdown("""Project 4: DeepRacer – PPO Agent Prepping for the F1 (Marshal Turner, Dec 2025) documents the development of a reinforcement learning agent for AWS DeepRacer to simulate F1-style racing across three tracks (reInvent2019-wide, reInvent2019, and Vegas) in three race modes: Time-Trial, Object-Avoidance, and Head-to-Head.
The agent uses a CNN + LIDAR encoder (≈1.7M parameters) to process stereo grayscale cameras, a colored front camera, and 64-beam LIDAR data. PPO was selected for its stability in continuous action spaces (steering/throttle) and resilience to volatile rewards. Training spanned ~80,000+ episodes total.
The core breakthrough was a rigorously engineered “No Mercy, No Exploits” reward function: all bonuses were capped, multiplicative cascades eliminated, and six specific crawling/zigzag/wall-hugging exploits were systematically killed. This replaced an earlier unstable reward design that caused policy collapses and local maxima.""", unsafe_allow_html= True)

st.markdown("""
<div style = 'text-align: center; font-size: 30px'>Initial Web-Scrape Data Extraction""", unsafe_allow_html=True)

st.markdown("""All the data is pulled from the official [Yelp Reviews](https://www.yelp.com/biz/bottega-louie-los-angeles?osq=Bottega+Louie%2Freviews) 
            page for Bottega Louie. My selenium web-scrape iteratively pulls the first 10,000 review descriptions and respective star ratings.""", unsafe_allow_html= True)

pfp = Image.open("images/ReinforcementLearningP/car.png")
st.image(pfp) 

pfp = Image.open("images/ReinforcementLearning/PPO.png")
st.image(pfp) 

pfp = Image.open("images/ReinforcementLearning/racetracks.png")
st.image(pfp) 

st.markdown("""Time-Trial: Reached 76.3% max progress and clean laps on the wide track (sub-16s potential shown in video).
Object-Avoidance: Max progress 26.3%; strong on straights but struggled with curve-based obstacles.
Head-to-Head: Competitive early but faded after ~25% due to turn-handling against 3 AI opponents.""", unsafe_allow_html= True)


st.markdown("""
<div style = 'text-align: center; font-size: 30px'>Project 3: Collaborative Onion Soup Delivery via QMIX with Dense Reward Shaping""", unsafe_allow_html=True)

st.markdown(""" (Marshal Turner, Nov 2025) trains two cooperative agents using QMIX to deliver at least 7 onion soups within 400 seconds across three Overcooked kitchen layouts: cramped room, coordination ring, and counter-circuit-o1order.
The agents receive a 96-feature observation vector and 6 discrete actions. QMIX was chosen for its monotonic value factorization (per-agent GRUs + state-conditioned mixing network) to enable decentralized execution while maintaining joint optimality in a loosely cooperative setting. A dense + event-based reward shaping function was layered on top of the environment rewards (onion/pot/dish/delivery bonuses + idle and collision penalties), scaled to ~10% of the base signal, with an episodic memory dictionary to block redundant actions and prevent loops.""", unsafe_allow_html= True)


col1, col2, col3 = st.columns([1,1,1])
with col1:
       pfp = Image.open("images/ReinforcementLearning/Map 1.png")
       st.image(pfp) 
with col2:
       pfp = Image.open("images/ReinforcementLearning/Map 2.png")
       st.image(pfp) 
with col3:
       pfp = Image.open("images/ReinforcementLearning/Map 3.png")
       st.image(pfp) 
       






