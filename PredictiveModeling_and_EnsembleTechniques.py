#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 10:39:33 2022

@author: Parth Patel
student no : 301027843
"""

import pandas as pd
import os
import numpy as np
 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Load & check the data:    
#--------------------------------------------------------------------------------------------------------------

path = "/Users/sambhu/Desktop/sem-2/Supervised learning 247/Assignment /Ass-4- Ensemble"
filename = 'pima-indians-diabetes.csv'
fullpath = os.path.join(path,filename)
df_parth = pd.read_csv(fullpath,sep=',')
 
df_parth.rename(columns = {'6'      : 'preg', 
                           '148'    : 'plas',
                           '72'     : 'pres',
                           '35'     : 'skin',
                           '0'      : 'test',
                           '33.6'   : 'mass',
                           '0.627'  : 'pedi',
                           '50'     : 'age',
                           '1'      : 'class'}, inplace = True)

print(df_parth.dtypes)
print(df_parth.isnull().sum())
print(df_parth.describe())
print(df_parth.median())
print(df_parth.columns.values)
print(df_parth.head(5))
print(df_parth.info())
print(df_parth['class'].value_counts())

#Pre-process and prepare the data for machine learning        
#--------------------------------------------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
transformer_parth = StandardScaler()

x=df_parth.iloc[:,0:8]
y=df_parth['class']


from sklearn.model_selection import train_test_split
X_train_parth,X_test_parth, y_train_parth, y_test_parth = train_test_split(x,y, test_size = 0.3, random_state=42)


X_train_parth_trn=transformer_parth.fit_transform(X_train_parth)
X_test_parth_trn=transformer_parth.fit_transform(X_test_parth)


X_train_parth_trn = pd.DataFrame(X_train_parth_trn, columns = ['preg','plas','pres','skin','test','mass','pedi','age'])
X_test_parth_trn = pd.DataFrame(X_test_parth_trn, columns = ['preg','plas','pres','skin','test','mass','pedi','age'])

#Exercise 1 :Hard voting           
#--------------------------------------------------------------------------------------------------------------
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier


log_clf = LogisticRegression(max_iter=1400)
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
dst_clf = DecisionTreeClassifier(criterion="entropy",max_depth = 42)
ext_clf = ExtraTreesClassifier()

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf),('dt', dst_clf),('et', ext_clf)],
    voting='hard')

voting_clf.fit(X_train_parth_trn, y_train_parth)

for clf in (log_clf, rnd_clf, svm_clf, dst_clf, ext_clf, voting_clf):
    clf.fit(X_train_parth_trn, y_train_parth)
    y_pred = clf.predict(X_test_parth_trn.head(3))
    print('--------------------------')
    print(clf.__class__.__name__,":\n--------------------------\n predicted result:", y_pred, "\n actual : \n",y_test_parth.head(3) )

#Exercise 2 :Soft voting           
#--------------------------------------------------------------------------------------------------------------

log_clf = LogisticRegression(max_iter=1400)
rnd_clf = RandomForestClassifier()
svm_clf_s = SVC( probability=True)
dst_clf = DecisionTreeClassifier(criterion="entropy",max_depth = 42)
ext_clf = ExtraTreesClassifier()

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf_s),('dt', dst_clf),('et', ext_clf)],
    voting='soft')

voting_clf.fit(X_train_parth_trn, y_train_parth)

for clf in (log_clf, rnd_clf, svm_clf, dst_clf, ext_clf, voting_clf):
    clf.fit(X_train_parth_trn, y_train_parth)
    y_pred = clf.predict(X_test_parth_trn.head(3))
    print('--------------------------')
    print(clf.__class__.__name__,":\n--------------------------\n predicted result:", y_pred, "\n actual : \n",y_test_parth.head(3) )

#Exercise #3: Random forests & Extra           
#--------------------------------------------------------------------------------------------------------------

from sklearn.pipeline import Pipeline

pipeline1_parth = Pipeline(
    steps=[("preprocessor", transformer_parth), ("ext", ext_clf)]
    )

pipeline2_parth = Pipeline(
    steps=[("preprocessor", transformer_parth), ("dst", dst_clf)]
    )

pipeline1_trained= pipeline1_parth.fit(X_train_parth, y_train_parth)
pipeline2_trained= pipeline2_parth.fit(X_train_parth, y_train_parth)

crossvalidation = KFold(n_splits=10, shuffle=True, random_state=42)
score1 = np.mean(cross_val_score(pipeline1_parth, X_train_parth, y_train_parth, scoring='accuracy', cv=crossvalidation ))
print("First pipeline mean score: ",score1)
score2 = np.mean(cross_val_score(pipeline2_parth, X_train_parth, y_train_parth, scoring='accuracy', cv=crossvalidation ))
print("Second pipeline mean score: ",score2)



y_pred1 = pipeline1_trained.predict(X_test_parth) 

y_pred2 = pipeline2_trained.predict(X_test_parth)

from sklearn import metrics
from sklearn.metrics import confusion_matrix
i=1
for pred in (y_pred1,y_pred2):
    if(i==1):
        print("---------------------------------------------------------")
        print("pipline1 with 'Extra Trees Classifier': ")
        print("---------------------------------------------------------") 
    else:
        print("---------------------------------------------------------")
        print("pipline2 with 'Decision  Trees Classifier': ")
        print("---------------------------------------------------------")
    print("\n\nAccuracy:",metrics.accuracy_score(y_test_parth, pred))
    print("Precision:",metrics.precision_score(y_test_parth, pred))
    print("Recall:",metrics.recall_score(y_test_parth, pred))
    print("f1 score:",metrics.f1_score(y_test_parth, pred) )
    
    CM=confusion_matrix(y_test_parth, pred)
    print("Confusion matrix :\n"  , CM)
    i=i+1
print(y_test_parth.value_counts())
# Exercise #4: Extra Trees and Grid search                  
#--------------------------------------------------------------------------------------------------------------
    
    
from sklearn.model_selection import RandomizedSearchCV
parameters=[{'ext__n_estimators' : range(10,3000,20),
            'ext__max_depth': range(1,1000,2)}]


rgs_43 = RandomizedSearchCV(estimator= pipeline1_parth,
                         scoring='accuracy',
                         param_distributions=parameters,
                         cv=5,
                         n_iter = 7,
                         refit = True,
                         verbose = 3)

rgs_43.fit(X_train_parth, y_train_parth)

print("best parameters:",rgs_43.best_params_)

print("Score",rgs_43.best_score_)

best_model = rgs_43.best_estimator_

rgs_pred = best_model.predict(X_test_parth)

print("recall:",metrics.recall_score(y_test_parth, rgs_pred))

print("precision:",metrics.precision_score(y_test_parth, rgs_pred))

print("Accuracy:",metrics.accuracy_score(y_test_parth, rgs_pred))

CM2=confusion_matrix(y_test_parth, rgs_pred)
print("Confusion matrix :\n"  , CM2)
   