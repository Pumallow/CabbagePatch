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

st.markdown("""<h1 style= 'text-align: center; font-size: 20px;'>Expand the sidebar on the left hand side to see my projects!</h1>""", unsafe_allow_html= True)
 
with open("images/4.29 Resume.pdf", "rb") as pdf_file:
    PDFByte = pdf_file.read()
       
scol1, scol2, scol3, scol4 = st.columns([5,1,1,5])
with scol2:
    st.download_button(label = "Resume", data = PDFByte, file_name = "4.29 Resume.pdf")         
with scol3:
    st.markdown("[LinkedIn](https://www.linkedin.com/in/mturner95/)", unsafe_allow_html= True)
   
intf = open(r"images/Imputation Methods： Uncovering Data Science's Hidden Magic!.mp4", 'rb')
intb = intf.read() 

video_html = """
<video controls width="250" autoplay="true" muted="true" loop="true">
<source 
            src="images/Imputation Methods： Uncovering Data Science's Hidden Magic!.mp4"
            type="video/mp4" />
</video>"""






width = 50
width = max(width, 0.01)
side = max((100 - width) / 2, 0.01)

col1, col2 = st.columns([5,2])
with col1: 
    st.markdown("""<h1 style= 'text-align: left; font-size: 20px;'> "Do not say what you'll do. Do what you say."
     </h1>""", unsafe_allow_html= True)
    st.markdown("""<h1 style= 'text-align: left; font-size: 20px;'> This website is dedicated to all of the studying/projects I commited to outside of work! 
      Passionate NLP enthusiast skilled in harnessing the power of data science, analytics, and engineering to unravel intricate patterns within datasets. 
      Adept at employing diverse programming languages and tools to uncover nuanced linguistic insights. 
      Demonstrated proficiency in crafting and deploying machine learning models tailored for text classification and sentiment analysis, driving impactful outcomes like reducing error rates 
      and enhancing revenue streams. Actively pursuing a Master's in Analytics to expand NLP expertise. Eager to leverage my knowledge and capabilities to propel a progressive team 
      within an innovative organization forward.
     </h1>""", unsafe_allow_html= True) 
    st.markdown("""<h1 style= 'text-align: left; font-size: 20px;'> In December of 2023, I acquired the GTx Micro Masters certificate. Currently, I am
      a graduate student at Georgia Tech in the Online Masters of Science in Analytics.
     </h1>""", unsafe_allow_html= True) 
    st.write("""<h1 style= 'text-align: center; font-size: 35px;'>Udemy Course Certifications </h1>
    <h1 style= 'text-align: left; font-size: 20px;'>The Complete SQL Bootcamp by Jose Portilla <br>
    Advanced SQL: SQL Expert Certification Preparation by Tim Buchalka <br>
    Become a SQL Developer (SSRS, SSIS,SSAS, T-SQL, DW) by BlueLime Learning Institution <br>
    Complete Python Bootcamp: Go from Zero to Hero in Python 3 by Jose Portilla <br>
    Alteryx Bootcamp <br>
    Snowflake Decoded - Fundamentals and hands on Training <br>
    Machine Learning A-Z: Hands-On Python & R in Data Science <br>
    Data Science in Layman’s Terms: Time Series Analysis <br>""", unsafe_allow_html= True)
    st.write("""<h1 style= 'text-align: center; font-size: 35px;'>Progress in Georgia Tech Masters of Science in Analytics</h1>
    <h1 style= 'text-align: left; font-size: 20px;'>SUMMER 23: Introduction to Analytics Modeling <br>
    FALL 23: Data Analytics for Business <br>
    FALL 23: Computing for Data Analysis <br>
    SPRING 24: Business Fundamentals for Analytics <br> 
    SPRING 24: Data Visuals & Analytics</h1>""", unsafe_allow_html= True)
    #st.markdown("<h1 style= 'text-align: left; color: white; font-size: 20px;'> A menu for all my past projects is located on the left hand side of the screen (extend it by arrow in the top left corner). </h1>", unsafe_allow_html= True)
    
    st.markdown("""<h1 style= 'text-align: left; font-size: 20px;'> My passion for data extends beyond my career with, for example, talking on podcasts. Outside of my career activities, I spend my time socializing with friends, playing soccer, working out, and playing the piano. I look to make the most
    out of life which mostly entails laughs and jokes! The video to the right is one of many things I've created to mess with my friends.</h1>""", unsafe_allow_html= True) 
    st.markdown("<h1 style= 'text-align: left; font-size: 20px;'>I hope to continually update this site with new content. Cheers! :D </h1>", unsafe_allow_html= True)

with col2:
  pfp = Image.open("images/PFP.JPG")
  st.image(pfp)
  st.video(intb)
  #st.markdown(video_html, unsafe_allow_html=True)

 



st.sidebar.success("Pages to Peruse.")
