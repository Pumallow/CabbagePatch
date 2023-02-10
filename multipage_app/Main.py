import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

st.set_page_config(layout="wide", page_title="Marshal's Data Science")
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; '>Marshal Turner's Data Science Website</h1>", unsafe_allow_html = True)
with open("images/Resume 2.10.2023.pdf", "rb") as pdf_file:
    PDFByte = pdf_file.read()
scol1, scol2, scol3, scol4 = st.columns([5,1,1,5])
with scol2:
    st.download_button(label = "Resume", data = PDFByte, file_name = "MarshalResume.pdf")         
with scol3:
    st.markdown("[LinkedIn](https://www.linkedin.com/in/mturner95/)", unsafe_allow_html= True)
    
vf =  open(r'images/troy.mov', 'rb')
vb = vf.read()

width = 50
width = max(width, 0.01)
side = max((100 - width) / 2, 0.01)

col1, col2 = st.columns([5,2])
with col1: 
    st.markdown("""<h1 style= 'text-align: left; font-size: 20px;'> This website is dedicated to all of the studying/projects I commited to outside of work! 
      I am a born and raised Georgian! Graduating from Kennesaw State & University with a degree in Industrial and Systems Engineering, I found myself with a fascination in tech.
      This ultimately led me to teach myself languages like SQL and Python as well as expose myself to software like Snowflake, Azure Databricks, Google BigQuery, PowerBi, Tableau, and more.
     </h1>""", unsafe_allow_html= True) 
    st.write("""<h1 style= 'text-align: left; font-size: 20px;'>Udemy Course Certifications <br>
    The Complete SQL Bootcamp by Jose Portilla, November 2019 <br>
    Advanced SQL: SQL Expert Certification Preparation by Tim Buchalka, January 2020 <br>
    Become a SQL Developer (SSRS, SSIS,SSAS, T-SQL, DW) by BlueLime Learning Institution, February 2021 <br>
    Complete Python Bootcamp: Go from Zero to Hero in Python 3 by Jose Portilla, February 2021 <br>
    Alteryx Bootcamp, July 2021 <br>
    Snowflake Decoded - Fundamentals and hands on Training, July 2021 <br>
    Machine Learning A-Z: Hands-On Python & R in Data Science April 2022 <br>
    Data Science in Layman’s Terms: Time Series Analysis September 2022</h1>""", unsafe_allow_html= True)
    #st.markdown("<h1 style= 'text-align: left; color: white; font-size: 20px;'> A menu for all my past projects is located on the left hand side of the screen (extend it by arrow in the top left corner). </h1>", unsafe_allow_html= True)
    st.markdown("<h1 style= 'text-align: left; font-size: 20px;'>I hope to continually update this site with new content. Cheers! :D </h1>", unsafe_allow_html= True)
with col2:
  pfp = Image.open("images/PFP.JPG")
  st.image(pfp)
col1, col2 = st.columns([2,5])
with col1:
  st.video(data=vb)
with col2:
  st.markdown("""<h1 style= 'text-align: left; font-size: 20px;'> Beyond the career drive, I spend my time elsewhere with friends or working on other activities such as soccer, working out, or playing the piano. I look to make the most
  out of life which mostly entails laughs and jokes! The video below is one of many things I've created to mess with my friends.</h1>""", unsafe_allow_html= True) 



st.sidebar.success("Pages to Peruse.")
