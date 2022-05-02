from tkinter import Grid
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

traindata = pd.read_csv(r'E:\DataScience\Titanic\train.csv')
ftest = pd.read_csv(r'E:\DataScience\Titanic\test.csv')
Titan = pd.concat([traindata, ftest], ignore_index= False)
ftest.info()
Titan.info()
Titan.describe()
#for i in Titan.columns:
    #plt.hist(Titan[i])
    #plt.title(i)
    #plt.show()
#38% of people survived 
#Most people were middle class 
#As expected there's an average age of 29 (adult) thus correlating with the parental/child and sib/spouse values.
#Highest age was 80 and lowest was a baby (0.5)
#Average fare price was $32 however it was quite dispersed with a $49 stand dev. and max of $512.....why pay $512 
Titan[Titan['Fare'] > 200]

#So only 3 people bought B Level Cabins at $512
#C Level Cabins come in around $250
#I am noticing that there is some type of trend with "PC" at the start of the Ticket for different individuals....let's dive deeper into it
#Why are there tickets with Letters in them?
Titan['Ticket'].unique()

Titan[Titan['Ticket'].astype(str).str[:2] == "PC"]
ttest = Titan.copy()
ttest['Ticket Type'] = np.where(ttest['Ticket'].str.contains(" "), ttest['Ticket'].str.split(' ').str[0], 'Normal')
ttest[ttest['Ticket Type'] != 'Normal']
#226 tickets that have some type of letter combination in front
ttest['Ticket Type'].unique()
#(All for Fun and Curiosity)
#After doing some background research I found that "Soton" could be an abbreviation for Southampton
#The abbreviated name of Southampton also is "C of Soton" and "SO"
#"SC" could be an abbreviation for Salford (Manchester)
#"A5" is a major road that runs across the UK (252 miles long) - in England and in Wales

#As much as I would like to further inspect this data I feel that theorizing what the abbreviations 
#could potentially mean is a bit too lofty for my comfort so we will move on.

#Does the Embark matter with Cabin Orientation?
cabtest = Titan.copy()
Cher = cabtest[cabtest['Embarked'] == 'C'] #Cherbourg
Cher['CClass'] = Cher['Cabin'].astype(str).str[0]
Cher['Number'] = Cher['Cabin'].astype(str).str[1:4]
Cher['Number'].unique()
Que = cabtest[cabtest['Embarked'] == 'Q'] #Queenstown
Sh = cabtest[cabtest['Embarked'] == 'S'] #Southampton
Cher.describe()
#69 ticket orders
#For Cherbourg average Fare is $60/PClass rounded is 1.88/Survival rate is 55%/Age is 30
Que.describe()
#4 ticket orders
#For Queenstown average Fare is $13.30/PClass rounded is 2.9/Survival rate is 50%/Age is 28  
Sh.describe()
#129 ticket orders
#For Cherbourg average Fare is $63/PClass rounded is 1/Survival rate is 62%/Age is 37


#Age, Cabin, and Embarked have null values
Titan['Parch'].unique()
Titan['SibSp'].unique()
#Can possibly divy up the train set by age. Look at the success of living based on adult vs child (18 +/-)
#Can look at children with parents vs divorced parents
Titan[Titan['Age'] >= 18].value_counts
# 601 of the passengers are considered "adult"
# Parents vs divorced would prove to be a small set to handle


#logical approach to replacing the NA in cabins --- multivariate imputer
cabinot = Titan[Titan['Cabin'].notna()] #204 records where Cabin is NA
cabinot[cabinot['Pclass']==1].value_counts #176 Class 1 --- ABCD
cabinot[cabinot['Pclass']==2].value_counts #16 Class 2 -- DEF
cabinot[cabinot['Pclass']==3].value_counts #12 Class 3 --- Cabins EFG
cabinot['CC'] = cabinot['Cabin'].astype(str).str[0]
cabinot

sns.barplot(x= 'CC', y= 'Pclass', data=cabinot)
plt.show()


#Now after initially analyzing the data let's do some cleaning
#1. I think it would be in my best interest to combine the SibSp or Parch....
#2. Let's take the age mean with the standard dev and randomly assign ages to NULL values. 
#3. Cabins is the tricky part. - we are cutting away the fat on the values for cabins by just keeping the class (A,B,C..)


pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 10)
Titan['Family'] = Titan['SibSp'] + Titan['Parch']
Titan['Cabin'] = Titan['Cabin'].astype(str).str[0]
Titan['Cabin'].replace(to_replace='n', value=np.NAN, inplace= True) 
#Now we gotta prep the Cabin for imputation
Titan['Cabin'].unique()
conditions = [Titan['Cabin'] == 'A', Titan['Cabin'] == 'B', Titan['Cabin'] == 'C', Titan['Cabin'] == 'D', Titan['Cabin'] == 'E', Titan['Cabin'] == 'F', Titan['Cabin'] == 'G', Titan['Cabin'] == 'T']
choices = [int(0),int(1),int(2), int(3),int(4),int(5), int(6),int(7)]
Titan['CabNum'] = np.select(conditions, choices, default=np.nan)
Titan['Embarked'].unique()
conditions = [Titan['Embarked'] == 'S', Titan['Embarked'] == 'C', Titan['Embarked'] == 'Q']
choices = [int(0),int(1),int(2)]
Titan['ENum'] = np.select(conditions, choices, default=np.nan)
Titan['Gen'] = np.where(Titan['Sex'] == 'male', 1, 0)
#You could use hotencoder however, I feel that'd make for too many columns
Titan1 = Titan.iloc[:, 1:13]
sns.pairplot(data= Titan1)
plt.show()


sns.pairplot(data= Titan1, x_vars= ["Fare"], y_vars= ["Family"], kind = 'reg', palette = 'cool warm')
plt.show()
#I am getting the impression that even though the column "Embarked" is insightful and interesting to look at, it also
#follows the nature of both fare and family. I believe I may leave it out.


#now to drop rows based on relevancy
#Dropping "Ticket" after assessing that there are not too many clear cut patterns with the numbers behind them
#Since we are using a multivariate imputer I need to drop the name
Titan.drop(columns= {'Sex','SibSp','Parch','Embarked','Ticket', 'Name','Cabin', 'ENum'}, inplace = True)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer as ii
imp = ii(missing_values = np.NaN, max_iter = 10, random_state = 0, initial_strategy= 'mean')
imp.fit(Titan.iloc[:, 2:8])
Titan.iloc[:, 2:8] = imp.transform(Titan.iloc[:, 2:8])

#HOWEVER before we get this show on the road remember that the CabNumbers can't exactly be 1.5654 lol
#So I gotta round them out.
Titan['CabNum'] = Titan['CabNum'].round(decimals= 0)

#Now that cabin is no longer a whole column of NULLs we can see if fare has a direct correlation to the cabin number.
sns.pairplot(data= Titan, x_vars = ["CabNum"], y_vars= ["Fare"], kind ='reg')
plt.show()


#It appears there's an expected downward trend of the pricing as you digress down the alphabet so I will remove the fare

Titan.drop(columns = {'Fare'}, inplace=True)
X_tr = Titan[Titan['PassengerId'] <= 891]
X_te = Titan[Titan['PassengerId'] > 891]
Titan.describe()
X_t1 = X_tr.iloc[:, 2:7]
Y_t1 = X_tr.iloc[:, 1:2]

ftest = X_te.iloc[:, 2:7]
ytest = X_te.iloc[:, 1:2]




from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_t1, Y_t1, test_size = 0.2, random_state = 1)


#Now that it is split we can standardize
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_final = sc.transform(ftest)
print(X_test)
print(X_final)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score

lr = LogisticRegression(max_iter= 2000, random_state= 0, penalty= "l2", C=0.0001, solver= 'liblinear')
lr.fit(X_train, Y_train)
y_pred = lr.predict(X_test)


cm = confusion_matrix(y_true = Y_test, y_pred = y_pred) 
lc = accuracy_score(y_true = Y_test, y_pred = y_pred)
pc = precision_score(y_true = Y_test, y_pred = y_pred)
rc = recall_score(y_true = Y_test, y_pred = y_pred)
f1 = f1_score(y_true = Y_test, y_pred = y_pred)

print("The confusion matrix for Logistic Regression: ", cm,"\r\n",
"The accuracy for Logistic Regression: ", lc,"\r\n",
"The precision for Logistic Regression: ", pc,"\r\n",
"The recall score for Logistic Regression: ", rc,"\r\n",
"The f1 score for Logistic Regression: ", f1,"\r\n")

knn = KNeighborsClassifier( algorithm= 'auto', n_neighbors= 120, weights= 'uniform')
knn.fit(X_train, Y_train)
knny_pred = knn.predict(X_test)
kcm = confusion_matrix(y_true = Y_test, y_pred = knny_pred) 
klc = accuracy_score(y_true = Y_test, y_pred = knny_pred)
kpc = precision_score(y_true = Y_test, y_pred = knny_pred)
krc = recall_score(y_true = Y_test, y_pred = knny_pred)
kf1 = f1_score(y_true = Y_test, y_pred = knny_pred)

print("The confusion matrix for KNN: ", kcm,"\r\n",
"The accuracy for KNN: ", klc,"\r\n",
"The precision for KNN: ", kpc,"\r\n",
"The recall score for KNN: ", krc,"\r\n",
"The f1 score for KNN: ", kf1,"\r\n")

svm = SVC(C= 10, gamma= 0.1, kernel= 'rbf') 
svm.fit(X_train, Y_train)
svmpred_y = svm.predict(X_test)
scm = confusion_matrix(y_true = Y_test, y_pred = svmpred_y) 
slc = accuracy_score(y_true = Y_test, y_pred = svmpred_y)
spc = precision_score(y_true = Y_test, y_pred = svmpred_y)
src = recall_score(y_true = Y_test, y_pred = svmpred_y)
sf1 = f1_score(y_true = Y_test, y_pred = svmpred_y)

print("The confusion matrix for SVM: ", scm,"\r\n",
"The accuracy for SVM: ", slc,"\r\n",
"The precision for SVM: ", spc,"\r\n",
"The recall score for SVM: ", src,"\r\n",
"The f1 score for SVM: ", f1,"\r\n")


#Right now, we are hitting around an 78.8% for accuracy with the KNN. Let's optimize it with the GridSearch
from sklearn.model_selection import GridSearchCV
svcparam_grid = {'C': [0.1, 1, 10, 100, 1000],
                 'gamma': [1,0.1,0.01, 0.001, 0.0001],
                 'kernel': ['rbf', 'sigmoid']}
optsvm = GridSearchCV(SVC(), param_grid= svcparam_grid, refit = True, verbose = 2)

svmp = optsvm.fit(X_train, Y_train)
print(
    "Best SVM Score is: ", svmp.best_score_,"\r\n",
    "Best Parameters for SVM is: ", svmp.best_params_, "\r\n",
    "Best Estimator for SVM is: ", svmp.best_estimator_)

sv2 = SVC(C= 10, gamma= 0.01, kernel= 'rbf') 
sv2.fit(X_train, Y_train)
fy = sv2.predict(X_test)
scfy = confusion_matrix(y_true = Y_test, y_pred = fy) 
slfy = accuracy_score(y_true = Y_test, y_pred = fy)
spfy = precision_score(y_true = Y_test, y_pred = fy)
srfy = recall_score(y_true = Y_test, y_pred = fy)
sfy = f1_score(y_true = Y_test, y_pred = fy)

print("The confusion matrix for SVM: ", scfy,"\r\n",
"The accuracy for SVM: ", slfy,"\r\n",
"The precision for SVM: ", spfy,"\r\n",
"The recall score for SVM: ", srfy,"\r\n",
"The f1 score for SVM: ", sfy,"\r\n")





knnparam_grid = {'n_neighbors' : [20,40,60,80],
                 'weights' : ['uniform', 'distance'],
                 'algorithm' : ['auto', 'ball_tree', 'kd_tree']}

optknn = GridSearchCV(KNeighborsClassifier(), param_grid= knnparam_grid, refit= True, verbose= 2, n_jobs= -1)
knnp = optknn.fit(X_train, Y_train)


print(
    "Best KNN Score is: ", knnp.best_score_,"\r\n",
    "Best Parameters for KNN is: ", knnp.best_params_, "\r\n",
    "Best Estimator for KNN is: ", knnp.best_estimator_)

knnfp = knnp.predict(X_test)
knfy = confusion_matrix(y_true = Y_test, y_pred = knnfp) 
knac = accuracy_score(y_true = Y_test, y_pred = knnfp)
knpc = precision_score(y_true = Y_test, y_pred = knnfp)
knrc = recall_score(y_true = Y_test, y_pred = knnfp)
knf1 = f1_score(y_true = Y_test, y_pred = knnfp)

print("The confusion matrix for KNN: ", knfy,"\r\n",
"The accuracy for KNN: ", knac,"\r\n",
"The precision for KNN: ", knpc,"\r\n",
"The recall score for KNN: ", knrc,"\r\n",
"The f1 score for KNN: ", knf1,"\r\n")


#No overfitting since the test set SVM model accuracy and the train set SVM model accuracy allign.

#Final Metrics:
#The confusion matrix for KNN:  [[94 12]
                                #[22 51]]
#The accuracy for KNN:  0.8100558659217877
#The precision for KNN:  0.8095238095238095
#The recall score for KNN:  0.6986301369863014
#The f1 score for KNN:  0.7500000000000001

knnt = KNeighborsClassifier( algorithm= 'auto', n_neighbors= 120, weights= 'uniform')
knnt.fit(X_train, Y_train)
test= sc.transform(ftest)
knnyt_pred = knn.predict(test)
affy = confusion_matrix(y_true = Y_test, y_pred = knnyt_pred) 
bffy = accuracy_score(y_true = Y_test, y_pred = knnyt_pred)
cffy = precision_score(y_true = Y_test, y_pred = knnyt_pred)
dffy = recall_score(y_true = Y_test, y_pred = knnyt_pred)
effy = f1_score(y_true = Y_test, y_pred = knnyt_pred)

print("The confusion matrix for KNN: ", affy,"\r\n",
"The accuracy for KNN: ", bffy,"\r\n",
"The precision for KNN: ", cffy,"\r\n",
"The recall score for KNN: ", dffy,"\r\n",
"The f1 score for KNN: ", effy,"\r\n")

lty = knn.predict(X_final)

pred_svm = pd.DataFrame({
        "PassengerId": X_te['PassengerId'],
        "Survived": lty.astype(int)
})
pred_svm.to_csv(r'E:\DataScience\Titanic\submission_svc.csv', index=False)

lt2 = sv2.predict(X_final)

pred_svm = pd.DataFrame({
        "PassengerId": X_te['PassengerId'],
        "Survived": lt2.astype(int)
})
pred_svm.to_csv(r'E:\DataScience\Titanic\SVMexperimentsubmission_svc.csv', index=False)