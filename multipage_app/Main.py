import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

showWarningOnDirectExecution = true

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>Marshal Turner's Data Science Website</h1>", unsafe_allow_html = True)
vf =  open(r'images/troy.mov', 'rb')
vb = vf.read()

width = 50
width = max(width, 0.01)
side = max((100 - width) / 2, 0.01)

col1, col2 = st.beta_columns([5,2])
with col1: 
    st.markdown("""<h1 style= 'text-align: left; color: white; font-size: 20px;'> Here are some works I have built along my journey as a data analyst and data scientist to prove/better my skills.
      I am a born and raised Georgian! Graduating from Kennesaw State & University with a degree in Industrial and Systems Engineering, I found myself with a fascination in tech.
      This ultimately led me to teach myself languages like SQL and Python as well as expose myself to software like Snowflake, Azure Databricks, Google BigQuery, PowerBi, Tableau, and more.
     </h1>""", unsafe_allow_html= True) 
    st.markdown("<h1 style= 'text-align: left; color: white; font-size: 20px;'> A menu for all my past projects is located on the left hand side of the screen (extend it by arrow in the top left corner). </h1>", unsafe_allow_html= True)
    st.markdown("<h1 style= 'text-align: left; color: white; font-size: 20px;'>I hope to continually update this site with new content. Cheers! :D </h1>", unsafe_allow_html= True)
    st.markdown("If you are interested in connecting with me, feel free to check out my [LinkedIn](https://www.linkedin.com/in/mturner95/).", unsafe_allow_html= True)
  
with col2:
  pfp = Image.open("images/PFP.JPG")
  st.image(pfp)
col1, col2 = st.beta_columns([2,5])
with col1:
  st.video(data=vb)
with col2:
  st.markdown("""<h1 style= 'text-align: left; color: white; font-size: 20px;'> Beyond the career drive, I spend my time elsewhere with friends or working on other activities such as soccer, working out, or playing the piano. I look to make the most
  out of life which mostly entails laughs and jokes! The video below is one of many things I've created to mess with my friends.</h1>""", unsafe_allow_html= True) 



st.sidebar.success("Pages to Peruse.")
