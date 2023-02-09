import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>Marshal Turner's Data Science Website</h1>", unsafe_allow_html = True)

col1, col2 = st.columns([1,1])
with col1:
  st.markdown(
    """
    <h1 style = 'text-align: left; color: white; font-size: 20px;'>
    Here are some works I have built along my journey as a data analyst and data scientist to prove/better my skills. \n
    I am a born and raised Georgian! Graduating from Kennesaw State & University with a degree in Industrial and Systems Engineering, I found myself with a fascination in tech.
    This ultimately led me to teach myself languages like SQL and Python as well as expose myself to software like Snowflake, Azure Databricks, Google BigQuery, PowerBi, Tableau, and more.
    Beyond the career drive, I spend my time elsewhere with friends or working on other activities such as soccer, working out, or playing the piano. I look to make the most
    out of life which mostly entails laughs and jokes! The video below is one of many things I've created to mess with my friends.
    
    I hope to continually update this site with new content. Cheers! :D
    </h1>""")
  st.markdown('If you are interested in learning more feel free to check out my [LinkedIn](https://www.linkedin.com/in/mturner95/).', unsafe_allow_html= True)
with col2:
  pfp = Image.open("images/PFP.JPG")
  st.image(pfp)
  
st.sidebar.success("Pages to Peruse.")
