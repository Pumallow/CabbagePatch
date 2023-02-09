

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import openpyxl


data = pd.read_excel(io= r'E:\DataScience\Data Science 2/TeamData.xlsx', sheet_name='FINAL', usecols= 'A:V', header = 1)
F1 = Image.open(r'E:\DataScience\Data Science 2\pics\France1.jpg')
K2 = Image.open(r'E:\DataScience\Data Science 2\pics\KSA2.jpg')
E3 = Image.open(r'E:\DataScience\Data Science 2\pics\England3.jpg')

w,h = E3.size
f = F1.resize((400,225))
k = K2.resize((400,225))
e = E3.resize((400,225))

st.set_page_config(layout="wide")
title_alignment="""
<style>
#the-title {
  text-align: center
}
</style>
"""
st.markdown(title_alignment, unsafe_allow_html=True)
st.title ("WHAT EPL TEAM SHOULD YOU SUPPORT?")
st.header("This quiz will deduce what team you should support this upcoming 23/24 season!")


col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.image(f, caption="You can dribble, cross, or shoot.")
    st.selectbox('How will you react to 1?',['Dribble','Shoot','Pass'])
with col2:
    st.image(k, caption="You are dribbling across the top of the box.")
    st.selectbox('How will you react to 2?',['Dribble','Shoot','Pass'])
with col3:
    st.image(e, caption="The ball has been headed to you.")
    st.selectbox('How will you react to 3?',['Dribble','Shoot','Pass'])

st.multiselect('Pick 2 Colors: ', ['Red','Yellow','Blue','Green', 'White'],max_selections=2)

st.selectbox('Which is most important for a team to have?', ['Defense', 'Offense', 'Both'])

st.selectbox('How much do you care for the tradition/history of a team?', ['Alot', 'Some', 'Not At All'])

st.selectbox('What types of games excite you most?', ['Upsets','Blow Outs','Close Games'])

st.selectbox('What do you think is most important for a club to invest in?', ['Local Soccer Academies', 'Scouting and Imported Talent', 'Fan Based Marketing'])

st.selectbox('What is the most important attribute for a winger?', ['Speed','Dribbling','Vision'])

st.selectbox('What is the most important attribute for your left and right backs?', ['Speed','Strength','Height'])

st.markdown('Once you have filled out all the questions above please click the "Submit button"')
st.button("Submit")



# 1. Which style of play is better? Offense, Defense, or Both? ---------------- DONE
# 2. How much do you care for tradition/history? Alot, some, not at all? ------------------DONE
# 3. Pick 2 colors: white, red, blue, green, yellow ---------------------DONE
# 4. Look at the situation: (picture).. would you pass, shoot, or dribble? ---------------------------DONE
# 5. What do you enjoy to spectate more? Upsets, Blow Outs, Close games?  ---------------------------DONE
# 6. What is most important for a team? Local Soccer Academies, Scouts and Imported Talent, marketing? -------------- DONE
# 7. What matters most to win? Finishing, Possession, Team Synergy? --------------- DONE
# 7. Which is best? Young Talent, Veteran,  