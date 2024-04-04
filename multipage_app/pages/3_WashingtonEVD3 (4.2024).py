import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import seaborn as sns
from matplotlib import pyplot as plt


st.set_page_config(layout="wide", page_title="D3 Washington Electric Vehicle Project")
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.markdown("""
<div style = 'text-align: center; font-size: 30px'>D3 JavaScript Geo-Projection for Washington EV Activity""", unsafe_allow_html=True)

intf = open(r"multipage_app/pages/d3EV/Final Video.mp4", 'rb')
intb = intf.read() 
st.video(data=intb)

st.markdown("""
<div style = 'text-align: center; font-size: 30px'>Evaluation of the Sales Activity Data""", unsafe_allow_html=True)

st.markdown("""<h1 style= 'text-align: left; font-size: 20px;'> When first looking at the [Kaggle Dataset](https://www.kaggle.com/datasets/willianoliveiragibin/electric-vehicle-population?resource=download), I wanted to perform
high level assessments the data. Through the use of panda tools like .info(), .describe(), and .value_counts() I illustrated these visuals
to show the top 10 most expensive EV models and most active cities.
</h1>""", unsafe_allow_html= True)

col1, col2, col3 = st.columns([1,1,1])
with col1:
       pfp = Image.open("multipage_app/pages/d3EV/City_MSRP.jpg")
       st.image(pfp) 
with col2:
       mp = Image.open("multipage_app/pages/d3EV/Model MSRP.jpg")
       st.image(mp) 
with col3:
       corr = Image.open("multipage_app/pages/d3EV/Correlation.jpg")
       st.image(corr) 

st.markdown("""<h1 style= 'text-align: left; font-size: 20px;'> To accommodate for JSON Geo-Projection for the D3 visual, I transitioned to a counties level of detail rather than cities. The charging stations are sourced from an
API tap on the [NREL Website](https://developer.nrel.gov/docs/transportation/alt-fuel-stations-v1/all/).
</h1>""", unsafe_allow_html= True)





