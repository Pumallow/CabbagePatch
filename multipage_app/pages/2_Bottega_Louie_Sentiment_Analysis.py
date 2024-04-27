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

st.markdown("""For our classifier to best consume each description, NLTK and Sklearn is used to simplify and remove fluff from the data entries. \n
1. First to go was punctuation. \n
2. Stopwords, words within the English language meant for grammar or phrasing but not specifically helpful with evaluating
sentiments, are filtered out. I.E. prepositional phrases, articles, or certain verbage \n
3. Groupings of root words and their conjugations are then made. I.E. [Running, Runs, Runner, Ran] -> run \n
\n
In essence, the filtering turns reviews into a latin-esque format.""", unsafe_allow_html= True)

col1, col2 = st.columns([1,1])
with col1:
       contain = st.container(height = 150, border=True)
       contain.markdown("""<div style = 'font-size: 20px'>Before the transformation:</div>
       <div>"Beautiful restaurant and always delicious food! I always enjoy the Carbonara- my favorite. But pizzas have been great as well as their Pomodoro pasta."</div>""", unsafe_allow_html=True) 
with col2:
       contain2 = st.container(height = 150, border=True)
       contain2.markdown("""<div style = 'font-size: 20px'>After the transformation:</div>
       <div>"beauti restaur alway delici food alway enjoy carbonara favorit pizza great well pomodoro pasta"</div>""", unsafe_allow_html= True)

st.markdown("""Between Bag-of-words model and Term Frequency - Inverse Document Frequency model, I chose a TF-IDF Vectorizer to perform the sentiment analysis. 
This type of vectorizer would allow for further manipulation with the kind of words evaluated. This was the initial model:""", unsafe_allow_html= True)

st.code("""from sklearn.feature_extraction.text import TfidfVectorizer as TfV
vector = TfV(max_features = 1500, min_df = 3, max_df = 0.6)
X = vector.fit_transform(root).toarray()""", language = "python")

st.markdown("""<div style = 'text-align: center; font-size: 20px'>To avoid overfitting to the star ratings, I decided to group the ratings into 3 buckets: <br><br>
Positive (4+ stars) <br>
Neutral (3 stars) <br>
Bad (3> stars)""", unsafe_allow_html= True)

st.markdown("""With the average rating settling on 4.1 stars, the expectation my K-Nearest-Neighbor classifier would label each description as positive was high. 
This quickly wsa confirmed with the first model and confusion matrix""", unsafe_allow_html= True)

st.markdown("""<div style = 'text-align: center; font-size: 30px'>First Test""", unsafe_allow_html=True)

st.markdown("KNN Accuracy: 77.49%", unsafe_allow_html=True)
test1 = Image.open("images/BottegaNLP/Test1.jpg")
st.image(test1) 






