import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

vf =  open(r'E:\DataScience\Data Science 2\pics\troy.mov', 'rb')
vb = vf.read()
pfp = Image.open(r'E:\DataScience\Data Science 2\pics\PFP.jpg')
arrow = Image.open(r'E:\DataScience\Data Science 2\pics\down arrow.npg')

w,h = pfp.size
p = pfp.resize((250,265))
a = arrow.resize((150,165))

st.set_page_config(layout="wide", page_title= "Data Analytics/Science Journey")
st.markdown("<h1 style= 'text-align: center; color: white; font-size: 40px' >Who is Marshal Turner?</h1>", unsafe_allow_html = True)
st.sidebar.success("Pages to Peruse")

col1, col2 = st.columns([1,1])
with col1:
    st.markdown(
    """
    <h1 style = 'text-align: left; color: white; font-size: 10px;'>
    I am a born and raised Georgian! Graduating from Kennesaw State & University with a degree in Industrial and Systems Engineering, I found myself with a fascination in tech.
    This ultimately led me to teach myself languages like SQL and Python as well as expose myself to software like Snowflake, Azure Databricks, Google BigQuery, PowerBi, Tableau, and more.
    Beyond the career drive, I spend my time elsewhere with friends or working on other activities such as soccer, working out, or playing the piano. I look to make the most
    out of life which mostly entails laughs and jokes! The video below is one of many things I've created to mess with my friends. </h1>
    """, unsafe_allow_html= True)
with col2: 
    st.image(p, 'Yo soy una persona.')

width = 50
width = max(width, 0.01)
side = max((100 - width) / 2, 0.01)

_, container, _ = st.columns([side, width, side])
container.video(data=vb)
