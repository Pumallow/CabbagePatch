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


intf = open(r"multipage_app/pages/d3EV/Final Video.mp4", 'rb')
intb = intf.read() 
st.video(data=intb)