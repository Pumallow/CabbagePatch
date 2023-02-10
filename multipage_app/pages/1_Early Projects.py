import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

with open("images/Adventure Works Challenge.pdf", "rb") as pdf_file:
    PDFByte = pdf_file.read()

alt = Image.open('images/Alteryx.jpg')


st.set_page_config(layout="wide", page_title= "Data Analytics/Science Journey")
st.markdown("<h1 style= 'text-align: center; color: white; font-size: 40px' >Projects that Segway into Data Analytics</h1>", unsafe_allow_html = True)
st.sidebar.success("Pages to Peruse")
st.markdown("""June, 2021 <br>
This project helped me land my first Data Analyst job at NTG. I combined my love for soccer with my fascination for data by web scraping 2400 + pages of data on the Premier League Website in python, grooming the gathered data in my local SQL host, then presenting an interactive Power BI Dashboard. Its purpose is to highlight the best left footed players/teams vs the best right footed players/teams.""", unsafe_allow_html= True)
st.markdown("[ETL Project](https://www.linkedin.com/feed/update/urn:li:activity:6821946098987401216/)", unsafe_allow_html= True)
st.markdown("""July, 2021 <br>
I took the ETL Project one step further by portraying the Transformation phases of the ETL Process with Alteryx after taking an introductory course.""", unsafe_allow_html= True)
st.image(alt)

st.markdown("""August, 2021 <br> 
At this point in my journey, SQL had been used quite abit however, I wanted to gain a more holistic understanding of the language to solidify my confidence with technical interviews. I thought making an Adventure Works SQL challenge worksheet would
be the solution. """, unsafe_allow_html= True)
st.button("Download", data = PDFByte)
