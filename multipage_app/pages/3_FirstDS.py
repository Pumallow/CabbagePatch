import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import openpyxl

pfp = Image.open(r'E:\DataScience\Data Science 2\pics\Code Pic.jpg')
avt = Image.open(r'E:\DataScience\Data Science 2\pics\AVT.jpg')
c1 = Image.open(r'E:\DataScience\Data Science 2\pics\Chart1.png')
c2 = Image.open(r'E:\DataScience\Data Science 2\pics\Chart2.png')
c3 = Image.open(r'E:\DataScience\Data Science 2\pics\Chart3.png')
c4 = Image.open(r'E:\DataScience\Data Science 2\pics\Chart4.png')
c5 = Image.open(r'E:\DataScience\Data Science 2\pics\Chart5.png')
dfd = Image.open(r'E:\DataScience\Data Science 2\pics\dfdescribe.png')
df = Image.open(r'E:\DataScience\Data Science 2\pics\df.png')
hc1 = Image.open(r'E:\DataScience\Data Science 2\pics\HC1.png')
hc2 = Image.open(r'E:\DataScience\Data Science 2\pics\HC2.png')
hc3 = Image.open(r'E:\DataScience\Data Science 2\pics\HC3.png')
hc4 = Image.open(r'E:\DataScience\Data Science 2\pics\HC4.png')
hd = Image.open(r'E:\DataScience\Data Science 2\pics\hdescribe.png')
ac1 = Image.open(r'E:\DataScience\Data Science 2\pics\AC1.png')
ac2 = Image.open(r'E:\DataScience\Data Science 2\pics\AC2.png')
ac3 = Image.open(r'E:\DataScience\Data Science 2\pics\AC3.png')
ac4 = Image.open(r'E:\DataScience\Data Science 2\pics\AC4.png')
ad = Image.open(r'E:\DataScience\Data Science 2\pics\awdescribe.png')
hmd = Image.open(r'E:\DataScience\Data Science 2\pics\hmdescribe.png')
lrcm = Image.open(r'E:\DataScience\Data Science 2\pics\lrcm.png')
kcm = Image.open(r'E:\DataScience\Data Science 2\pics\knncm.png')
scm = Image.open(r'E:\DataScience\Data Science 2\pics\svmcm.png')
#rel = Image.open(r'E:\DataScience\Data Science 2\pics\relations.png')


w,h = pfp.size
p = pfp.resize((270,245))
a = avt.resize((250,264))

st.set_page_config(layout="wide", page_title= "Data Analytics/Science Journey")
st.markdown("""<h1 style= 'text-align: center; color: white; font-size: 40px'>To what extent does attendance and location effect the score for an Arsenal vs Tottenham match?</h1>""", unsafe_allow_html = True) 
st.sidebar.success("Pages to Peruse")
st.image(avt)
st.markdown(
    """
    <h1 style = 'text-align: left; color: white; font-size: 15px;'> 
     My first Data Science Project.
     After completing the Machine Learning A-Z: Hands-On Python & R In Data Science Udemy course, this project served as a display of my understandings from the course. It entails Data Extraction (selenium webscraping),
     Exploratory Data Analysis, AI Modeling (Logistic Regressions, SVM, KNN), and model optimization (GridSearchCV). 
     For context, Arsenal and Tottenham are 2 English Premier League soccer teams that have been notable rivals for many years. Whenever they play another, people call the match "The North London Darby" for it being such 
     a spectacle. I am a die hard Arsenal fan.
    </h1>
    """, unsafe_allow_html= True)   

st.markdown('The data pulled comes from an [Arsenal vs Tottenham](http://www.mehstg.com/arsestat.htm) Website. The Webscraping code is quite extensive. Click Button Below to expand the code block. (Refresh to contract)', unsafe_allow_html= True)


if st.button("WebScrape"):
    st.code("""
    def stats():
        global attable
        driver.get('http://www.mehstg.com/arsestat.htm')
        for r in range(2, 266):
            m = []
            for p in range(1, 8):
                try:
                    try:
                        test2 =  driver.find_element_by_xpath('/html/body/div[1]/center/table/tbody/tr[' + str(r) + ']/td[' + str(p) + ']/b[2]/font/br').text
                        if "aet" in test2: 
                            value4 = driver.find_element_by_xpath('/html/body/div[1]/center/table/tbody/tr[' + str(r) + ']/td[' + str(p) + ']/b[1]/font').text + ' - ' + driver.find_element_by_xpath('/html/body/div[1]/center/table/tbody/tr[' + str(r) + ']/td[' + str(p) + ']/b[2]/font/text()[1]').text
                            m.append(value4)
                            continue
                    except:
                        try:
                            test3 =  driver.find_element_by_xpath('/html/body/div[1]/center/table/tbody/tr[' + str(r) + ']/td[' + str(p) + ']/b/font/br').text
                            if test3 != None: 
                                value8 = driver.find_element_by_xpath('/html/body/div[1]/center/table/tbody/tr[' + str(r) + ']/td[' + str(p) + ']/b/font/text()[1]').text()
                                l = value8.split('"\"')[0]
                                m.append(value8)
                                continue
                        except:
                            test3 = 1
                    try:
                        value = driver.find_element_by_xpath('/html/body/div[1]/center/table/tbody/tr[' + str(r) + ']/td[' + str(p) + ']/font/b').text
                        if (value == "-" or value == " -") and p == 4: 
                            value1 = driver.find_element_by_xpath('/html/body/div[1]/center/table/tbody/tr[' + str(r) + ']/td[' + str(p) + ']/b[1]/font').text + ' - ' + driver.find_element_by_xpath('/html/body/div[1]/center/table/tbody/tr[' + str(r) + ']/td[' + str(p) + ']/b[2]/font').text 
                            m.append(value1) 
                        elif "&nbsp" in value:
                            if value[0].isdigit():
                                t2 = value[0] + " - " + value[-1]
                                m.append(t2)
                        elif value.startswith(" -"):
                            value3 = driver.find_element_by_xpath('/html/body/div[1]/center/table/tbody/tr[' + str(r) + ']/td[' + str(p) + ']/b/font').text + driver.find_element_by_xpath('/html/body/div[1]/center/table/tbody/tr[' + str(r) + ']/td[' + str(p) + ']/font/b').text
                            m.append(value3)
                        elif value == "&nbsp;-&nbsp;":
                            t1 = driver.find_element_by_xpath('/html/body/div[1]/center/table/tbody/tr[' + str(r) + ']/td[' + str(p) + ']/b[2]/font').text
                            if "(a.e.t.)" in t1:
                                value3 = driver.find_element_by_xpath('/html/body/div[1]/center/table/tbody/tr[' + str(r) + ']/td[' + str(p) + ']/b[1]/font').text + ' - ' + driver.find_element_by_xpath('/html/body/div[1]/center/table/tbody/tr[' + str(r) + ']/td[' + str(p) + ']/b[2]/font').text
                                value5 = value3.split("(")[0]
                                m.append(value5)
                                continue
                            value1 = driver.find_element_by_xpath('/html/body/div[1]/center/table/tbody/tr[' + str(r) + ']/td[' + str(p) + ']/b[1]/font').text + ' - ' + driver.find_element_by_xpath('/html/body/div[1]/center/table/tbody/tr[' + str(r) + ']/td[' + str(p) + ']/b[2]/font').text 
                            m.append(value1) 
                        elif value.endswith("-"):
                            value6 = driver.find_element_by_xpath('/html/body/div[1]/center/table/tbody/tr[' + str(r) + ']/td[' + str(p) + ']/font/b').text + " " + driver.find_element_by_xpath('/html/body/div[1]/center/table/tbody/tr[' + str(r) + ']/td[' + str(p) + ']/b/font').text 
                            m.append(value6)
                        elif "A" == value[0]:
                            m.append("A")
                        elif "H" == value[0]:
                            m.append("H")
                        elif "Wembley" == value or "Stamford Bridge" == value or "Park Royal" == value or "Old Trafford" == value:
                            m.append("H")
                        elif value != None:
                            m.append(value)
                    except:
                        value2 = driver.find_element_by_xpath('/html/body/div[1]/center/table/tbody/tr[' + str(r) + ']/td[' + str(p) + ']/b/font').text
                        if "Covid" in value2:
                            continue
                        if value2 == "??":
                            m.append("None")
                        elif "A" == value2[0]:
                            m.append("A")
                        elif "H" == value2[0]:
                            m.append("H")
                        elif "Wembley" == value2 or "Stamford Bridge" == value2 or "Park Royal" == value2:
                            m.append("H")
                        else:
                            m.append(value2)
                except:
                    m.append("None")
                    #/html/body/div[1]/center/table/tbody/tr[35]/td[4]/b[2]/font/br


            lo = pd.Series(m, index = attable.columns)
            attable = attable.append(lo, ignore_index = True)
""",language= "python")
    
st.markdown(
    """
    After successfully scraping the futbol data, I chose to define the scope of my extraction by Attendance, Location, Spurs Score, Arsenal Score,
    and a numerical representation of who won or if there was a tie per game. (0 = Tottenham Spurs Win / 1 = Arsenal Win / 2 = Tie) 
    """, unsafe_allow_html= True)
    


st.code("""
conditions = [attable['Spurs Score'] > attable['Arsenal Score'], attable['Spurs Score'] < attable['Arsenal Score'], attable['Spurs Score'] == attable['Arsenal Score']]
choices = [int(0),int(1),int(2)]
#0 means that the Spurs won
#1 means that Arsenal Won
#2 means that they tied
attable['Winner'] = np.select(conditions, choices, default=np.nan)
attable['Attendance'] = attable['Attendance'].str.replace(',','')
attable['Attendance'] = attable['Attendance'].astype(float)
print(attable)
    """)

col1, col2 = st.columns([1,1])
with col1:
    st.image(df)
with col2:
    st.write("""
    "Attable" serves as my parent table for any exploratory data analysis. I attempted to convert all relative attributes into numerical values to later normalize the data for modeling.
    From a high level, we see that Arsenal won 97 times, Spurs have won 84 times, and the 2 teams tied each other 62 times thus far for a total of 243 North London Darby's played.
    There is an expectancy for what we should see in the trend of attendence. Emirates and Tottenah HotSpurs stadium seem to host similar numbers of attendance per game with the Emirates Stadium 
    at a max capacity of 60,361 fans and Tottenham HotSpurs Stadium ahead with 62,850.
    """)  
st.code("""
attable.describe()
""",language= "python")


col1, col2 = st.columns([1,1])
with col1:
    st.image(dfd)
with col2:    
    st.write(
        """
        After taking a close look at the table description for our scores and attendance we see that:
        The average attendance is 37K people.
        On average, Spurs score around 1.403 points per game.
        On average, Arsenal scores around 1.497 points per game.
        The "Location" stats are based off of a numerical representation of if the game was hosted at home being (Tottenham stadium - 100) vs away (Arsenal Emirates Stadium - 90). This will be the basis of how I later split the parent table.
        With the "Winner" stats we must acknowledg the results are different due to the tie value of 2 throwing it off.
        """, unsafe_allow_html= True)      


st.code("""
import matplotlib.pyplot as plt
for i in attable.columns:
    plt.hist(attable[i])
    plt.title(i)
    if i == "Attendance":
        plt.xlabel("Attendance Recorded Per Game")
    if i == "Location":
        plt.xlabel("Tottenham = 100 / Arsenal = 90")
    if i == "Spurs Score" or i == "Arsenal Score":
        plt.xlabel("Goals Made Per Game")
    if i == "Winner":
        plt.xlabel("Spurs Win = 0 / Arsenal Win = 1 / Tie = 2")
    plt.ylabel("Game Occurences")
    plt.show()
""",language= "python")

col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.image(c1)
    st.image(c2)
with col2:
    st.image(c3)
    st.image(c4)
with col3:
    st.image(c5)
    st.markdown(
        """<h1 style = 'text-align: left; color: white; font-size: 15px;'> We can see that attendance has a healthy bell curve shape (disregarding influx of lower attendance).
        The wins, losses, and ties are not far apart by any means. Spurs have won around 20 less than Arsenal (THAT'S BECAUSE ARSENAL IS BETTER) however, they've tied quite a bit. The trend for both Spurs and Arsenal scores and Location show to be normal.  
        </h1>
        """, unsafe_allow_html= True) 

st.code("""
attable.info()
attable.sort_values('Attendance',axis=0,ascending = True, inplace= True,kind='stable')
print(attable)
#Let us also create 2 additional tables that highlight only home games and only away games
htable = attable[attable['Location'] == 100]
awtable = attable[attable['Location'] == 90]

htable.describe()
""",language= "python")

st.markdown(
    """<h1 style = 'text-align: left; color: white; font-size: 15px;'> The parent table is then split by the location for each game (Home = "htable" / Away = "awtable") to further analyze
    the discrepancies of each team's performance. This is to statistically solidify the notions behind a team having the "home advantage". </h1>
    <h1 style = 'text-align: center; color: white; font-size: 30px;'> Shown below is a description of the "Tottenham Stadium" dataset. </h1>
    """, unsafe_allow_html= True) 

col1, col2 = st.columns([1,1])
with col1:
    st.image(hd)
with col2:
    st.markdown(
    """<h1 style = 'text-align: left; color: white; font-size: 15px;'> First things first, attendance falls below the parent average of 37K to 35K despite the HotSpur's stadium larger size and the standard deviation remains relatively stagnant around 19.5K. As expected, we see the average score of Spurs rise from 1.4 to 1.6 and the Arsenal
     average score fall from 1.49 to 1.33. Following behind those 2 metrics is the "Winner" average falling from 0.9 to 0.81 (remember 0 represents Spurs winning). </h1>
    """, unsafe_allow_html= True) 

col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.image(hc1)
    st.image(hc2)
with col2:
    st.image(hc3)
    st.image(hc4)
with col3:
    st.markdown(
    """<h1 style = 'text-align: left; color: white; font-size: 15px;'> We can see that attendance has some what of a health bell curve shape (disregarding influx of lower attendance).
    It is interesting to see that the wins, losses, and ties are not far apart by any means. 
    Spurs have won around 20 less than Arsenal (THAT'S BECAUSE ARSENAL IS BETTER) however, they've tied quite a bit. The trend for both Spurs and Arsenal scores and Location show normal trends.  
    </h1>
    """, unsafe_allow_html= True) 


st.markdown(
    """
    <h1 style = 'text-align: center; color: white; font-size: 30px;'> Shown below is a description of the "Emirates Stadium" dataset. </h1>
    """, unsafe_allow_html= True)

col1, col2 = st.columns([1,1])
with col1:
    st.image(ad)
with col2:
    st.markdown(
    """<h1 style = 'text-align: left; color: white; font-size: 15px;'> In comparison to the overall performances, the attendance of the Emirates stadium exceeds the overall average attendance of Tottenham stadium with an average
    attendance of 2,500 more (40,000). Arsenal goals per game average increase by 0.18 and Spurs score drop by 0.22 however the standard deviations for both vary little. 
    </h1>
    """, unsafe_allow_html= True)  

col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.image(ac1)
    st.image(ac2)
with col2:
    st.image(ac3)
    st.image(ac4)
with col3:
    st.markdown(
    """<h1 style = 'text-align: left; color: white; font-size: 15px;'> Emirates stadium seats an average of 5,000 more than the Tottenham stadium. The more surprising portion are the scores. There is a difference in performance for
    home vs away games for Spurs by nearly 0.5 goals per game as opposed to 0.3 goals per game performance by Arsenal. Based on the aggregate metrics and the Spurs chart one can argue that if the Spurs beat Arsenal in Emirates that the margin 
    would only by a deficit of 1 majority of the time. Further, with there being a lower standard deviation on the Winner column for Emirates stadium, speculations can be made that Arsenal do not play
    as well as the Spurs do at home but compensate by playing better Away. </h1>
    """, unsafe_allow_html= True) 


st.markdown(
    """
    <h1 style = 'text-align: center; color: white; font-size: 30px;'> Visual Depiction of Relationships between each Attribute </h1>
    """, unsafe_allow_html= True)

col1, col2 = st.columns([1,1])
with col1: 
    st.image(pfp)
with col2:
    st.image(hmd)
    st.markdown(
    """<h1 style = 'text-align: left; color: white; font-size: 15px;'> Using the seaborn heatmap, we actually can see the EXTENT of correlation between attendance and winning being -0.1231 which is not too serious but still impactful relationship.
    The relationship between attendance and location is shown to be strong (-0.12629). All other stated factors are supported by the visual. The Spurs Score and Location have a good relationship given they are a "home advantage" team as opposed to 
    Arsenal who has a bit less of a correlation between their perforamnce with where they play. On the other hand, Arsenal do play better with larger number of fans attending. One can argue that the rise in attendance at an away game for Arsenal was 
    contributed by the die hard Arsenal fans travelling with the team to the Tottenham Stadium. </h1>
    """, unsafe_allow_html= True) 

st.markdown(
    """
    <h1 style = 'text-align: center; color: white; font-size: 30px;'> Modeling the Data </h1>
    """, unsafe_allow_html= True)

st.markdown(
    """
    <h1 style = 'text-align: left; color: white; font-size: 20px;'> Step 1: Normalize the Data </h1>
    """, unsafe_allow_html= True)

st.code("""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dtable = attable.copy()
dtable.drop(columns = {"Spurs Score", "Arsenal Score"}, inplace = True)
X = dtable.iloc[:, :-1].values
Y = dtable.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 1)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
""",language= "python")

st.markdown(
    """
    <h1 style = 'text-align: left; color: white; font-size: 20px;'> Step 2: Structure the Models </h1>
    """, unsafe_allow_html= True)

st.write("I was a bit overwhelmed by the level of detail you could dive into with each model so I decided to apply grid searches and thin my selection options.")

st.code("""
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np

svc = SVC()
lr = LogisticRegression()
knn = KNeighborsClassifier()

lrparam_grid = {'max_iter': [2000],
              'penalty': ['l2'],
              'C': np.logspace(-4, 4, 20),
              'solver': ['liblinear']}

knnparam_grid = {'n_neighbors' : [78,80,82,84], #had to mess around with numbers to see what was suggested. Eventually landed on 80
                 'weights' : ['uniform', 'distance'],
                 'algorithm' : ['auto', 'ball_tree', 'kd_tree']}
                 
svcparam_grid = {'C': [0.1, 1, 10, 100],
                 'gamma': [1,0.1,0.01, 0.001],
                 'kernel': ['rbf','poly','sigmoid']}


optlr = GridSearchCV(lr, param_grid = lrparam_grid, cv = 5, verbose = True, n_jobs = -1 )
optknn = GridSearchCV(knn, param_grid= knnparam_grid, cv = 5, verbose = True, n_jobs = -1)
optsvm = GridSearchCV(svc, param_grid= svcparam_grid, refit = True, verbose = 2)
lrp = optlr.fit(X_train, Y_train)
knnp = optknn.fit(X_train, Y_train)
svmp = optsvm.fit(X_train, Y_train)
""",language= "python")

st.markdown('''
Best Logist Regression Score is:  0.4671171171171172 (46.7%) \n
 Best Parameters for Logist Regression is:  {'C': 0.0001, 'max_iter': 2000, 'penalty': 'l2', 'solver': 'liblinear'} \n
 Best KNN Score is:  0.47822822822822825 (47.8%) \n 
 Best Parameters for KNN is:  {'algorithm': 'auto', 'n_neighbors': 80, 'weights': 'uniform'} \n
 Best SVM Score is:  0.46726726726726725 (46.7%) \n
 Best Parameters for SVM is:  {'C': 0.1, 'gamma': 1, 'kernel': 'poly'}''')

st.markdown(
    """
    <h1 style = 'text-align: left; color: white; font-size: 20px;'> Step 3: Test the Modelss </h1>
    """, unsafe_allow_html= True)

st.markdown(  """
    <h1 style = 'text-align: center; color: white; font-size: 20px;'> Logistic Regression Modeling </h1>
    """, unsafe_allow_html= True)

st.code("""
lr = LogisticRegression(max_iter= 2000, random_state= 0,penalty= "l2", C=0.0001, solver= 'liblinear')
lr.fit(X_train, Y_train)
print(lr.predict(sc.transform([[20000,100]])))
#It is predicted that the Spurs will win if a game that is played at home and the attendance is 20,000 
print(lr.predict(sc.transform([[40000,100]])))
#It is predicted that the Spurs will win if a game that is played at home and the attendance is 40,000 
print(lr.predict(sc.transform([[20000, 90]])))
#It is predicted that Arsenal will win if a game is played away and the attendance is 20,000
print(lr.predict(sc.transform([[40000, 90]])))
#It is predicted that Arsenal will win if a game is played away and the attendance is 40,000
""",language= "python")


st.code("""
y_pred = lr.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_true = Y_test, y_pred = y_pred) 
lc = accuracy_score(y_true = Y_test, y_pred = y_pred)

print(cm)
print(lc)
""",language= "python")

col1, col2 = st.columns([1,1])
with col1:     
    st.markdown('''
    First trial was a basic model with a 25% test size and concluded with an accuracy score of 44.26% (just shy of best score being 46.7%). \n
    12 correct predictions for if Arsenal won. \n
    15 correct predictions if Tottenham won. \n
    So far, no surprises have been uncovered and I am concerned for overfitting since 25% of 263 records is around 65.''')
with col2: 
    st.image(lrcm)

st.markdown(  """
    <h1 style = 'text-align: center; color: white; font-size: 20px;'> KNN Modeling </h1>
    """, unsafe_allow_html= True)

st.code("""
knn = KNeighborsClassifier( algorithm= 'auto', n_neighbors= 80, weights= 'uniform')
knn.fit(X_train, Y_train)

knny_pred = knn.predict(X_test)
print(np.concatenate((knny_pred.reshape(len(knny_pred),1), Y_test.reshape(len(Y_test),1)),1))

cm = confusion_matrix(y_true = Y_test, y_pred = knny_pred) 
lc = accuracy_score(y_true = Y_test, y_pred = knny_pred)

print(cm)
print(lc)
""",language= "python")

col1, col2 = st.columns([1,1])
with col1:     
    st.markdown('''
    Got a bit lower of a score than the logistic regression with a 42.6% accuracy score and its best score being higher with a 47.8%. \n
    12 correct predictions for if Arsenal won. \n
    14 correct predictions if Tottenham won \n
    Really not too much of a difference. \n
    It's as expected that the GridSearch would not overhaul the expected outcome however, it does optimize within reason.''')
with col2: 
    st.image(kcm)

st.markdown(  """
    <h1 style = 'text-align: center; color: white; font-size: 20px;'> Support Vector Modeling </h1>
    """, unsafe_allow_html= True)

st.code("""
svm = SVC(C= 0.1, gamma= 1 , kernel= 'poly')
svm.fit(X_train, Y_train)

svmpred_y = svm.predict(X_test)
print(np.concatenate((svmpred_y.reshape(len(svmpred_y),1), Y_test.reshape(len(Y_test),1)),1))

cm = confusion_matrix(y_true = Y_test, y_pred = svmpred_y) 
lc = accuracy_score(y_true = Y_test, y_pred = svmpred_y)

print(cm)
print(lc)
""",language= "python")

col1, col2 = st.columns([1,1])
with col1:     
    st.markdown('''
    Lastly, SVM got the highest score with a 44.3% accuracy score and its best score being higher with a 47.8%. \n
    14 correct predictions for if Arsenal won. \n
    13 correct predictions if Tottenham won''')
with col2: 
    st.image(scm)

st.markdown(
    """
    <h1 style = 'text-align: center; color: white; font-size: 30px;'> Conclusion </h1>
    """, unsafe_allow_html= True)

st.write("""
This was my very first attempt at adhering AI models to a random set of data. 

In the end SVM yielded the highest accuracy score despite the best score provided by GridSearchCV! \n
Given the arbitrary nature of my webscraped data I am not surprised SVM shined given its ability to encompass a relatively linear set of data (as seen with the metrics description).
I am aware I could have "refit" the data with the exact suggested paramater tuning GridSearchCV provided which could have potentially turned the tables in favor of KNN.
Given the overall accuracy of all 3 models it seems there isn't too much of a patternistic behavior to the history of the North London Derby (Tottenham vs Arsenal)
Ideally, logistic regressions should provide something between a 60 - 70% to be considered "decent results" for a random set of data.

This project taught me many things from the true breakdown cycle of data from start to finish to understanding the nuances of parameter tuning with GridSearchCV.
I appreciate being able to see the use of simple histograms with analyzing data at a relatively basic level and spot insightful information on correlations with variables like location and attendance! \n
More specific examples are: \n
Arsenal won 97 times / Spurs won 84 times / they tied 62 times \n
Spur Home Games - +0.203 more goals scored for / -0.159 less goals scored against \n
Spur Away Games - -0.2223 less goals scored for / +0.1745 more goals scored against \n

There's actually a podcast I listened to that theorized it is (slightly) more difficult for a home team to win given they know the people in the stands.
This set of data seemed to prove otherwise. \n

So to return to my previously asked questions: \n
To what extent does location and attendance impact score? \n
Based on the multiple predictions made with our logistic regression attendance and location effect the outcome of games however, it is based on the outlook of the team. As stated before, Tottenham plays a lot better at home than
away in comparison to Arsenal however, Arsenal play better in general especially when more people are in the stadium. \n

Is "homefield advantage" an accurate statement? \n
Based on our data set "Homefield Advantage" IS an accurate statement! Tottenham Spurs without a doubt take advantage whenever playing home!  \n

Possible Errors: \n
 - Flaws in my parameter tuning \n
 - I am not sure if I was REQUIRED to use pipeline to avoid effecting other sets of data. I do not believe it did. \n
 - Given the poor nature of the accuracy rate, I didn't feel like I had enough self esteem at that point to torture myself with those numbers haha. \n
 - Hopefully, next time I can find a set of data with a bit more of a clustered nature or patternistic structure. \n

Possible Project Expansions:
 - If I were to expand on this project I would calculate the Error Rate, F1, and recall for all 3. \n
  - I would also backtrack and create a new set for a randomforestclassifier and see if there's a possibility of predicting scores given a set of parameters. \n

 With Tottenham's surprising ability to beat Man City this season it shows there really is no predictability in futbol HAHAHA""")
