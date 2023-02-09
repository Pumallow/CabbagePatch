import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>Marshal Turner's Data Science Website</h1>", unsafe_allow_html = True)

col1, col2 = st.columns([1,1])
with col1:
  st.write(
    """
    Here are some works I have built along my journey as a data analyst and data scientist to prove/better my skills.
    I hope to continually update this site with new content. Cheers :D
    """)
  st.markdown('If you are interested in learning more feel free to check out my [LinkedIn](https://www.linkedin.com/in/mturner95/).', unsafe_allow_html= True)
with col2:
  pfp = Image.open("images/PFP.JPG")
  st.image(pfp)
  
st.sidebar.success("Pages to Peruse.")
