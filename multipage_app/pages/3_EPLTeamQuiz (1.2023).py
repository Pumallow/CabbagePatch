import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import tree 
from sklearn.preprocessing import StandardScaler
import time
import os
from sklearn.neighbors import KNeighborsClassifier as KNN
F1 = Image.open('images/France1.JPG')
K2 = Image.open('images/KSA2.JPG')
E3 = Image.open('images/England3.JPG')
DTI = Image.open('images/DTInstability.png')
IA = Image.open('images/ImpurityvAlpha.png')

w,h = E3.size
f = F1.resize((400,225))
k = K2.resize((400,225))
e = E3.resize((400,225))

st.set_page_config(layout="wide")
st.markdown("""
<div style = 'text-align: center; font-size: 50px'>WHICH EPL TEAM SHOULD YOU SUPPORT?</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style = 'text-align: center; font-size: 30px'>This quiz will deduce what team you should support this upcoming 23/24 season!</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.image(f, caption="You can dribble, cross, or shoot.")
    first = st.selectbox('How will you react to 1?',['Dribble','Shoot','Pass'])
with col2:
    st.image(k, caption="You are dribbling across the top of the box.")
    second = st.selectbox('How will you react to 2?',['Dribble','Shoot','Pass'])
with col3:
    st.image(e, caption="The ball has been headed to you.")
    third = st.selectbox('How will you react to 3?',['Dribble','Shoot','Pass'])
    
fourth = st.selectbox("Which formation style is better?",['Offensive','Defensive','A Balance'])

fifth = st.selectbox('How much do you care about the tradition/history of a club?',['A lot','Some','Little'])

sixth = st.multiselect('Pick a Primary color, then a Secondary color: ',['Red','Blue','Yellow','Black','White','Maroon'],max_selections = 2)

seventh = st.selectbox('What type of game do you prefer to spectate the most?', ['Upsets','Blowouts','Close Games'])

eighth = st.selectbox('How would you like to recruit your team?', ['Local Soccer Academies','Scouting and Imported Talent'])

ninth = st.selectbox('Which is more important to you?', ['A large fanbase','A wealthy club', 'A High Trophy Count'])

twelvth = st.slider('What is a good average age for the players on your EPL team?',min_value = 24, max_value= 28, value = 26)

def gather():
    l = ''
    for i in sixth:
        l += i + "'"
    input = {
        'firstscen': first,
        'secondscen': second,
        'thirdscen': third,
        'style': fourth,
        'history': fifth,
        'color':  l,
        'spectate': seventh,
        'growth': eighth,
        'fan': ninth,
        'age': twelvth
    }
    # 1. Let's first format the input
    #input = {'firstscen': 'Shoot', 'secondscen': 'Dribble', 'thirdscen': 'Dribble', 'style': 'Defensive', 'history': 'A lot', 'color': "Black'Yellow'", 'spectate': 'Blowouts', 'growth': 'Local Soccer Academies', 'fan': 'A wealthy club', 'play': 'tiki-taka (quick, clean passing)', 'age': 24}
    data = pd.read_excel('td.xlsx', sheet_name='FINAL', usecols= 'A:Y', header = 1)
    inp = pd.DataFrame()
    values = [1,2,3]
    v2 = [1,2,3,4,5,6]
    fin = pd.concat([inp, pd.DataFrame.from_dict(input, orient= 'index').T])
    firstc = (fin['firstscen'] == 'Shoot',fin['firstscen'] == 'Dribble',fin['firstscen'] =='Pass')
    secondc = (fin['secondscen'] == 'Shoot',fin['secondscen'] == 'Dribble',fin['secondscen'] =='Pass')
    thirdc = (fin['thirdscen'] == 'Shoot',fin['thirdscen'] == 'Dribble',fin['thirdscen'] =='Pass')
    sty = (fin['style'] == 'Offensive',fin['style'] == 'Defensive',fin['style'] =='A Balance')
    hist = (fin['history'] == 'A lot',fin['history'] == 'Some',fin['history'] =='Little')
    spec = (fin['spectate'] == 'Upsets',fin['spectate'] == 'Blowouts',fin['spectate'] =='Close Games')
    fan = (fin['fan'] == 'A large fanbase', fin['fan'] == 'A wealthy club', fin['fan'] == 'A High Trophy Count')
    pc = (fin['color'].astype(str).str.split("'")[0][0] == "Red", fin['color'].astype(str).str.split("'")[0][0] == "Blue", fin['color'].astype(str).str.split("'")[0][0] == "Yellow", fin['color'].astype(str).str.split("'")[0][0] == "Black", fin['color'].astype(str).str.split("'")[0][0] == "White", fin['color'].astype(str).str.split("'")[0][0] == "Maroon")
    sc = (fin['color'].astype(str).str.split("'")[0][1] == "Red", fin['color'].astype(str).str.split("'")[0][1] == "Blue", fin['color'].astype(str).str.split("'")[0][1] == "Yellow", fin['color'].astype(str).str.split("'")[0][1] == "Black", fin['color'].astype(str).str.split("'")[0][1] == "White", fin['color'].astype(str).str.split("'")[0][1] == "Maroon")
    fin['fs'] = np.select(firstc, values, default=np.nan)
    fin['ss'] = np.select(secondc, values, default=np.nan)
    fin['ts'] = np.select(thirdc, values, default=np.nan)
    fin['x'] = fin['fs'] + fin['ss'] + fin['ts']
    fin['formation'] = np.select(sty, values, default=np.nan)
    fin['history'] = np.select(hist, values, default=np.nan)
    fin['spectate'] = np.select(spec, values, default=np.nan)
    fin['growth'] = np.where(fin['growth'] == 'Local Soccer Academies', 1, 0)
    fin['primary color'] = np.select(pc, v2, default=np.nan)
    fin['secondary color'] = np.select(sc, v2, default=np.nan)
    fin['fan'] = np.select(fan, values, default= np.nan)
    x = ((fin['x'] < 5), (fin['x'] > 7))
    fin['scenario'] = np.select(x, [1,3], 2)
    fin.drop(columns = ['firstscen','secondscen','thirdscen','style','fs','ss','ts','color','x'], inplace = True)
    fin.rename(columns = {'age':'Age'},inplace= True)
    fin = fin[['formation', 'primary color', 'secondary color','history','fan','spectate','scenario','growth','Age']]
    #Data Preprocessing
    dx = data.copy() 
    dx.drop(columns = {'Formation'}, inplace = True)
    dx['Touches'] = pd.to_numeric(dx['Touches'],downcast = 'float')
    dx['Passes'] = pd.to_numeric(dx['Passes'],downcast = 'float')
    dx['Shots'] = pd.to_numeric(dx['Shots'],downcast = 'float')
    #Making Calculative aggregates to determine if a team plays a passing, shooting, or dribbling centric style.
    #Whichever attribute has the highest percentage of total actions in a league in comparison to their other attributes will take that on as their "centric style"
    dx['PD'] = dx['Passes'].div(dx['Passes'].sum(), axis= 0)
    dx['TD'] = dx['Touches'].div(dx['Touches'].sum(), axis= 0) 
    dx['SD'] = dx['Shots'].div(dx['Shots'].sum(), axis= 0)
    dx['We'] = dx['Wealth'].div(dx['Wealth'].sum(), axis= 0)
    dx['TC'] = dx['Trophy Count'].div(dx['Trophy Count'].sum(), axis= 0)
    dx['FB'] = dx['Fan Base'].div(dx['Fan Base'].sum(), axis= 0)
    ipc = (dx['PC'] == "Red", dx['PC'] == "Blue", dx['PC'] == "Yellow", dx['PC'] == "Black", dx['PC'] == "White", dx['PC'] == "Maroon")
    isc = (dx['SC'] == "Red", dx['SC'] == "Blue", dx['SC'] == "Yellow", dx['SC'] == "Black", dx['SC'] == "White", dx['SC'] == "Maroon")
    wl = ((dx['Place'] > 13) , (dx['Place'] < 8), ((dx['Place'] < 14) & (dx['Place'] > 7)))
    history = ((dx['Create Year'] <= 1884), (dx['Create Year'] > 1884))
    form = ((dx['Style'] == "High Press"), (dx['Style'] == "Defensive"), (dx['Style'] == "Mid"))
    scen = (((dx['SD'] > dx['TD']) & (dx['PD'] < dx['SD'])), ((dx['PD'] < dx['TD']) & (dx['TD'] > dx['SD'])), ((dx['PD'] > dx['TD']) & (dx['PD'] > dx['SD'])))
    dfan = (((dx['FB'] > dx['TC']) & (dx['We'] < dx['FB'])), ((dx['We'] > dx['TC']) & (dx['We'] > dx['FB'])))
    dx['primary color'] = np.select(ipc, v2, default=np.nan)
    dx['secondary color'] = np.select(isc, v2, default=np.nan)
    dx['formation'] = np.select(form, values, default= np.nan)
    dx['history'] = np.select(history, [1,3], default = 2)
    dx['fan'] = np.select(dfan, [1,2], default = 3)
    dx['spectate'] = np.select(wl, values, default= np.nan)
    dx['scenario'] = np.select(scen, values, default= np.nan)
    dx['growth'] = np.where(dx['HomeGrown'] > 0.095, 1, 0)
    #dx.drop(columns = {'Unnamed: 0', 'Team','Style','PC','SC', 'Offsides', 'Create Year', 'Wealth', 'Fan Base', 'Trophy Count', 'TD','SD','PD','Passes','Touches','Shots', 'HomeGrown','FB', 'TC', 'We'}, inplace = True)
    dx = dx[['Team','formation', 'primary color', 'secondary color','history','fan','spectate','scenario','growth','Age']]
    dx['Age'] = dx['Age'].round()
    #For fun lets do some analysis
    
    # sns.heatmap(dx.corr())
    # plt.show() 
    # print(dx.corr())
    
    X_test = fin.values
    X = dx.iloc[:, 1:].values
    Y = dx.iloc[:, :1].values.ravel()


    classifier = KNN(n_neighbors= 20, weights = 'distance', algorithm= 'kd_tree', leaf_size= 100, p = 2)
    model = classifier.fit(X,Y)
    y_pred = model.predict(X_test)
    pred = list(y_pred)
    df = data[data['Team'].isin(pred)]
    return df, dx, fin

# Euclidean Distance	Mostly used for quantitative data
# Taxicab Geometry	Used when the data types are heterogenous
# Minkowski distance	Intended for real-valued vector spaces
# Jaccard index	Often used in applications when dealing with binarized data
# Hamming distance	Typically used with data transmitted over computer networks. And also used with categorical variables.


#Age == 25.68 / 26.8

#Defensive:
#least goals scored on 17 (bottom 25%)
#Style == 2 Defensive
#Cleansheets == 4.25 (top 75%) 4 (50%)
#Saves  == 267.25 (top 75%) 248.5 (50%)

#Offensive:
#most goals scored for 25.75 (top 75%)
#Style  == 1 Highpress

#Offsides == 27 (top 75%)

#Homegrown ==   0.184 (top 75%)

 
if st.button("Submit"):
    x, y, z = gather()
    with st.spinner('Wait for it...'):
        time.sleep(3)
    st.success('Done!')
    te = '' 
    tea = te.join(x['Team'].values) 
    team = tea + ".png"
    files = os.listdir('team logos/')
    for file in files:
        if file == team:
            logo = Image.open(os.path.join('team logos/', file))    
    plc = 0
    w = 0
    d = 0
    l = 0
    gf = 0
    ga = 0
    age = 0
    p = 0
    t = 0
    s = 0
    cs = 0
    y = 0
    save = 0 
    form = ''
    cy = 0 
    fb = 0
    sty = ''
    place = plc + x['Place'].iloc[0]
    creyear = cy + x['Create Year'].iloc[0]
    fanb = fb + x['Fan Base'].iloc[0]
    win = w + x['W'].iloc[0]
    draw = d + x['D'].iloc[0]
    lose = l + x['L'].iloc[0]
    gf = gf + x['GF'].iloc[0]
    ga = ga + x['GA'].iloc[0]
    age = age + x['Age'].iloc[0]
    passes = p + x['Passes'].iloc[0]
    touch = t + x['Touches'].iloc[0]
    shots = s + x['Shots'].iloc[0]
    cleansheet = cs + x['Cleansheets'].iloc[0]
    Yellow = y + x['Yellows'].iloc[0]
    saves = save + x['Saves'].iloc[0]
    style = sty.join(x['Style'].values)
    formation = form.join(x['Formation'].values)
    col1, col2 = st.columns([1,1])
    with col1:
        st.image(logo)
    with col2:
        st.markdown(
        """<h1 style = 'text-align: left;font-size: 20px;'>Congratulations! Based on your quiz results, for the English Premier League 23/24 season you should support {}! Founded in {}, the team has amassed a fan base of over {} (in millions).
        Last season, {} , came in {} place in the EPL! At the mid point of the 22/23 season, the team tallied {} passes, {} touches, and {} shots. On defense, {} had {} yellows however, they all contributed to {} saves and {} cleansheets!
        They scored a total of {} goals however were scored on {} times. 
        The team regularly runs a {} style of play with a {} formation.</h1>
        """.format(tea, creyear, fanb, tea, place, passes, touch, shots, tea, Yellow, saves, cleansheet, gf, ga, style, formation), unsafe_allow_html= True)
    st.markdown(
    """
    <h1 style = 'text-align: center; font-size: 30px;'>
    Go {}!!!!! </h1>
    """.format(tea), unsafe_allow_html= True)
    

if st.button("How Does it Work?"):
    try:
        x, y, z = gather()
        data = pd.read_excel('td.xlsx', sheet_name='FINAL', usecols= 'A:Y', header = 1)
        data.drop(columns = ['Unnamed: 0'], inplace = True)
        st.markdown(
        """
        <h1 style = 'text-align: center; font-size: 30px;'> Extracting the Data </h1>
        """, unsafe_allow_html= True)
        st.markdown(
        """
        <h1 style = 'text-align: left; font-size: 15px;'> 
        To successfully acquire the necessary data set shown below, I pulled from various sources across the internet on the latest EPL futbol stats. To best approach the task of building a "futbol team chooser quiz", I 
        first thought of attributes in a team I think should pivotally describe someone's preferances as a spectator. The quiz needs to highlight over not only the team's style on the field but their culture, history, fanbase, brand, and other factors that
        directly impact any fan's experience.   
        </h1>
        """, unsafe_allow_html= True)
        st.dataframe(data, width=1000, height=300)  
        st.markdown(
        """
        <h1 style = 'text-align: center; font-size: 30px;'> Preprocessing the Data </h1>
        """, unsafe_allow_html= True)   
        st.markdown(
        """
        <h1 style = 'text-align: left; font-size: 15px;'> 
        Once the data was structured and cleaned, an immediate obstacle stood out amongst the rest: any AI model would be given an initial bias based on the performance of top tier teams like
        Manchester City or Arsenal. As a result, any quiz result would end up being 1 of 3 responses rather than any 20. By converting a team's attributes to a "Percentage of the aggregate total" then
        comparing one to another, I siloed each team's performance to highlight their most popular activities on/off the field and easily differentiate them from another.
        </h1>
        """, unsafe_allow_html= True)
        st.code(
        """#These are the coded calculations for "Percentage of Aggregate Totals"
        dx['PD'] = dx['Passes'].div(dx['Passes'].sum(), axis= 0)
        dx['TD'] = dx['Touches'].div(dx['Touches'].sum(), axis= 0) 
        dx['SD'] = dx['Shots'].div(dx['Shots'].sum(), axis= 0)
        dx['We'] = dx['Wealth'].div(dx['Wealth'].sum(), axis= 0)
        dx['TC'] = dx['Trophy Count'].div(dx['Trophy Count'].sum(), axis= 0)
        dx['FB'] = dx['Fan Base'].div(dx['Fan Base'].sum(), axis= 0)""",language= "python")
        st.markdown(
        """
        <h1 style = 'text-align: left; font-size: 15px;'> 
        The rest of my preprocessing included various selection statements to format all the "facets" of a futbol fan's experience down to 3 numbers (1,2,3).
        </h1>
        """, unsafe_allow_html= True)
        st.code("""
        #Here is 1 Example:
        values = [1,2,3]
        form = ((dx['Style'] == "High Press"), (dx['Style'] == "Defensive"), (dx['Style'] == "Mid"))
        dx['formation'] = np.select(form, values, default= np.nan)
        """, language= "python")
        st.markdown(
        """
        <h1 style = 'text-align: center; font-size: 30px;'> Analyzing the Data </h1>
        """, unsafe_allow_html= True)
        st.markdown(
        """
        <h1 style = 'text-align: left; font-size: 15px;'> 
        Taking a closer look at the relations between categorical attributes we see that the "Formation" and "fan", "Age" and "primary color", and "Age" and "fan" share close bonds.
        Seeing this heat map settled my nerves because I initially thought the correlations between only fan, history, and growth would control the classification model. 
        </h1>
        """, unsafe_allow_html= True)
        plot = sns.heatmap(y.corr())
        st.pyplot(plot.get_figure())
        st.markdown(
        """
        <h1 style = 'text-align: center; font-size: 30px;'> Modeling the Data </h1>
        """, unsafe_allow_html= True)
        st.markdown(
                """
                <h1 style = 'text-align: left; font-size: 15px;'> 
                This later motivated me to attempt using a scikit Decision Tree Classifier model.
                After a bit of analysis however, I concluded that the Decision Tree would not perform as well as a Kernel Nearest Neighbor model. 
                A healthy model would show that the CCP (cost_complexity_pruning_path) would gradually decline in nodes over alphas until 1 remains. 
                Even with custom weighting however, my model would drop in nodes in a sudden fashion (i.e. a single question essentially dictates the decision trees answer).
                </h1>
                """, unsafe_allow_html= True)
        col1, col2 = st.columns([1,1])
        with col1:
            st.image(DTI)
        with col2:
            st.image(IA)
        st.markdown(
                """
                <h1 style = 'text-align: left; font-size: 15px;'> 
                The KNN model easily molded to my data set with the numerical categories I developed to "represent" each facet of a futbol fan's experience.
                I modified the weights to rely a KD-tree with a euclidean distance given it was a multi-variable dataset with few rows.
                </h1>
                """, unsafe_allow_html= True)
        st.code("classifier = KNN(n_neighbors= 20, weights = 'distance', algorithm= 'kd_tree', leaf_size= 100, p = 2)", language= "python")
        st.markdown(
        """
        <h1 style = 'text-align: center; font-size: 30px;'> Conclusion </h1>
        """, unsafe_allow_html= True)    
        st.markdown(
                """
                <h1 style = 'text-align: left; font-size: 15px;'> 
                Brainstorming questions to correctly develop a usable response from the user that then would be fed into a model proved to be the most difficult obstacle of this project.
                My largest take aways came from the intense dive I made into decision tree modeling and statistical theory behind KD-tree vs Ball Tree. This project emphasized the growing pressures
                to know how statistical theorums operate with preprocessed datasets given there is no established metric for accuracy or recall to fall back on for a quiz like this. 
                </h1>
                """, unsafe_allow_html= True) 
    except:
        st.write("Please take quiz before clicking this button :D")






# 1. Which style of play is better? Offense, Defense, or Both? D
# 2. How much do you care for tradition/history? Alot, some, not at all? D
# 3. Pick 2 colors: white, red, blue, green, yellow D
# 4. Look at the situation: (picture).. would you pass, shoot, or dribble? D
# 5. What do you enjoy to spectate more? Upsets, Blow Outs, Close games?   
# 6. What is most important for a team? Local Soccer Academies, Scouts and Imported Talent, marketing?
# 7. What matters most to win? Finishing, Possession, Team Synergy?
