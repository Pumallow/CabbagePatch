import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import openpyxl

vf =  open(r'E:\DataScience\Data Science 2\pics\Capstone ETL Project.mp4', 'rb')
vb = vf.read()
pfp = Image.open(r'E:\DataScience\Data Science 2\pics\PFP.jpg')

w,h = pfp.size
p = pfp.resize((250,265))

st.set_page_config(layout="wide", page_title= "Data Analytics/Science Journey")
st.markdown("<h1 style= 'text-align: center; color: white; font-size: 40px' >ETL Project - Segway into Data Analytics</h1>", unsafe_allow_html = True)
st.sidebar.success("Pages to Peruse")

st.markdown(
    "This project helped me land my first Data Analyst job at NTG. I combined my love for soccer with my fascination for data by web scraping 2400 + pages of data on the [Premier League Website] (https://www.premierleague.com/stats/top/players/goals?se=79) in python, grooming the gathered data in my local SQL host, then presenting an interactive Power BI Dashboard. Its purpose is to highlight the best left footed players/teams vs the best right footed players/teams.", unsafe_allow_html= True)

width = 80
width = max(width, 0.01)
side = max((100 - width) / 2, 0.01)

_, container, _ = st.columns([side, width, side])
container.video(data=vb)