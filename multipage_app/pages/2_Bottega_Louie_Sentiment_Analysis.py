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

st.markdown("""Bottega Louie, a gourmet restaurant located in Los Angeles, California, holds the record for most
            reviews on Yelp with 18,756 reviews. Each review contains a written description and rating of 
            1 to 5 stars. Bottega Louie holds an elite customer review average of 4.1 stars. Using NLP methodologies, can
            the descriptions of each review correctly be classified to predict the star rating?""", unsafe_allow_html= True)

st.markdown("""
<div style = 'text-align: center; font-size: 30px'>Initial Web-Scrape Data Extraction""", unsafe_allow_html=True)

st.markdown("""All the data is pulled from the official [Yelp Reviews](https://www.yelp.com/biz/bottega-louie-los-angeles?osq=Bottega+Louie%2Freviews) 
            page for Bottega Louie. My selenium web-scrape iteratively pulls the first 10,000 review descriptions and respective star ratings.""", unsafe_allow_html= True)

pfp = Image.open("images/BottegaNLP/Reviews.jpg")
st.image(pfp) 

st.markdown("""For our classifier to best consume each description, NLTK and Sklearn are used to simplify and remove fluff from the data entries. \n
1. Punctuation is first removed. \n
2. Stopwords, words within the English language meant for grammar or phrasing but not specifically helpful with evaluating
sentiments, are then filtered out. \n
\t I.E. prepositional phrases, articles, or certain verbage \n
3. Groupings of root words and their conjugations are then made. \n
\t I.E. [Running, Runs, Runner, Ran] -> run \n
\n
In essence, the filtering turns reviews into a latin-esque format:""", unsafe_allow_html= True)

col1, col2 = st.columns([1,1])
with col1:
       contain = st.container(height = 150, border=True)
       contain.markdown("""<div style = 'font-size: 20px'>Before the transformation:</div>
       <div>"Beautiful restaurant and always delicious food! I always enjoy the Carbonara- my favorite. But pizzas have been great as well as their Pomodoro pasta."</div>""", unsafe_allow_html=True) 
with col2:
       contain2 = st.container(height = 150, border=True)
       contain2.markdown("""<div style = 'font-size: 20px'>After the transformation:</div>
       <div>"beauti restaur alway delici food alway enjoy carbonara favorit pizza great well pomodoro pasta"</div>""", unsafe_allow_html= True)

st.markdown("""Between a Bag-of-words model and Term Frequency - Inverse Document Frequency model, I chose a TF-IDF Vectorizer to perform the sentiment analysis. 
A TF-IDF model would allow for further manipulation with the text. After iteratively testing the variables with a KNN classifier, this was the final vectorizer:""", unsafe_allow_html= True)

st.code("""from sklearn.feature_extraction.text import TfidfVectorizer as TfV
vector = TfV(max_features = 2000, min_df = 5, max_df = 0.5)
X = vector.fit_transform(root).toarray()""", language = "python")

st.markdown("""<div style = 'text-align: center; font-size: 20px'>To avoid overfitting to the star ratings, I decided to group the ratings into 3 buckets: <br><br>
Positive (4+ stars) <br>
Neutral (3 stars) <br>
Bad (3 > stars) <br> </div>""", unsafe_allow_html= True)

cont = st.container(border=True)
cont.markdown("""<div style = 'text-align: center; font-size: 30px'> Initial Observations: <br> </div> 
<div style = 'text-align: left; font-size: 20px'>From a high level, positive descriptions average shorter lengths than Neutral or Negative messages (Positive - 645 / Neutral - 841 / Negative - 844). <br>
Psychologically, humans don't respond as heavily to positive experiences as they do negative experiences. Positive descriptions are less anecdotal and as a result, turn out to be more 
consistent with diction and phrasing. <br>
On the other hand, negative experiences are anecdotal with more comments to provide illustrations thus creating variance in the text. <br>
I chose to add an additional "Neutal" bucket to assist the vectorizer with bucketing ambiguous descriptions. <br> 
With the average rating settling of 4.1 stars, I fully expectated my K-Nearest-Neighbor classifier would label each description as positive. 
This quickly was confirmed with the first model and confusion matrix. </div>""", unsafe_allow_html= True)

st.markdown("""<div style = 'text-align: center; font-size: 30px'>KNN Classifier""", unsafe_allow_html=True)

nn = Image.open("images/BottegaNLP/N_neighbors.jpg")
nnt = Image.open("images/BottegaNLP/nntest.jpg")

col1, col2 = st.columns([1,1])
with col1:
       st.image(nn)
with col2:
       st.markdown("I iterated through various number of neighbors included in each query to optimize the accuracy. Something to take note of however, the reason the accuracy increases with n_neighbors = 16 is because the model leans to optimize ONLY Positive comments. Lowering n_neighbors increases the accuracies for Neutral/Negative reviews. As a result, I stuck with n_neighbors = 5 to keep a decent overall metric while compensating for separate accuracies.", unsafe_allow_html= True)
       st.image(nnt)

test1 = Image.open("images/BottegaNLP/Test1.jpg")
st.image(test1) 

st.markdown("""<div style = 'text-align: center; font-size: 30px'>Grid Search""", unsafe_allow_html=True)
gcv = Image.open("images/BottegaNLP/GS.jpg")
fin = Image.open("images/BottegaNLP/Fintest.jpg")
col1, col2 = st.columns([1,1])
with col1:
       st.markdown("To best service the issue, the GridSearchCV needed to be scored with a 'balancing_accuracy'.", unsafe_allow_html= True)
       st.code("grid = GridSearchCV(knn, param_grid, cv=10, scoring='balanced_accuracy', return_train_score=False)", language = "python")
       st.markdown("After inputting the optimum metrics into the KNN classifier. I yielded the final results:", unsafe_allow_html= True)
       st.image(fin)      
with col2:
       st.image(gcv)

st.markdown("""<div style = 'text-align: center; font-size: 30px'>Conclusion""", unsafe_allow_html=True)
st.markdown("""Based on the sentimental analysis, the positive yelp comments are overwhelmingly easier to classify as opposed to negative. The neutral comments
are left to interpretation. The final results held a 71.83% overall accuracy only to accomodate a strict "3 Bucket" grade system for the 5 star reviews. In an ideal world
of only looking at Positive vs Negative comments, comparing the frequency for the use of a word to the totals document word count would serve as a viable classifier that pushes an accuracy of 85%+.
Another reason for why I added an extra "Neutral" bucket was to limit test our TF-IDF vectorizer + KNN combination.""", unsafe_allow_html=True)
st.markdown("""Potential Improvements:
1. Resample our webscrape for a more even pull of Positive/Neutral/Negative reviews.
2. Change the logic for the bucketing of Y_Actual (Positive/Neutral/Negative).
3. Test alternative vectorizers like FastText, Word2Vec, or OneHotEncoding.
4. Try an ANN with Tensorflow.""", unsafe_allow_html= True)








