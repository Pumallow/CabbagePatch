import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


st.set_page_config(layout="wide", page_title="NLP Project")
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.markdown("""
<div style = 'text-align: center; font-size: 30px'>Sentiment Analysis on Bottega Louie Yelp Reviews""", unsafe_allow_html=True)

p = Image.open("images/BottegaNLP/Bottega.jpg")
bot = st.image(p)

st.markdown("""Bottega Louie, a gourmet market restaurant located in Los Angeles, California holds the record for most
            reviews on Yelp with 18,756 reviews. Each review contains a written description and rating of 
            1 to 5 stars. Bottega Louie holds an elite customer review average of 4.1 stars. Using NLP methodologies, can
            the descriptions of each review correctly be classified to predict the star rating?""", unsafe_allow_html= True)

st.markdown("""
<div style = 'text-align: center; font-size: 30px'>Initial Web-Scrape Data Extraction""", unsafe_allow_html=True)

st.markdown("""All the data is pulled from the official [Yelp Reviews](https://www.yelp.com/biz/bottega-louie-los-angeles?osq=Bottega+Louie%2Freviews) 
            page for Bottega Louie. My selenium web-scrape iteratively pulled the first 10,000 review descriptions and respective star rating.""", unsafe_allow_html= True)

pfp = Image.open("images/BottegaNLP/Reviews.jpg")
st.image(pfp) 

st.markdown("""For our classifier to best consume each description, NLTK and Sklearn was used to simplify and remove fluff from the data entries. \n
1. First to go need to be punctuation. \n
2. Stopwords, words within the English language meant for grammar or phrasing but not specifically helpful with evaluating
sentiments, are filtered out. I.E. prepositional phrases, articles, or certain verbage \n
3. Groupings of root words and their conjugations are then made. I.E. [Running, Runs, Runner, Ran] -> run \n
\n
In essence, the filtering created this transformation.""", unsafe_allow_html= True)

trans = Image.open("images/BottegaNLP/Transformation.jpg")
st.image(trans) 



