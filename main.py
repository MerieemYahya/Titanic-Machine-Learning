import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


st.title('Binary Classification WebApp')
st.write("Results")

# Same treatment as in Jupeter nootbook

# Import data
df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')

# drop duplicate value
df_train = df_train.drop_duplicates()

# Fill null value

df_train['Age'].fillna(df_train['Age'].mode()[0], inplace=True)
df_train['Cabin'].fillna(df_train['Cabin'].mode()[0], inplace=True)
df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)

# Preprocessing data
df_train = df_train.drop(['PassengerId'], axis=1)

df_train['Title'] = df_train['Name'].str.extract('([A-Za-z]+)\.',)

df_train = df_train.drop(['Name'], axis=1)
df_train['Fare_Band'] = pd.cut(df_train['Fare'], 3)

df_train.loc[(df_train['Fare'] <= 170.776), 'Fare'] = 1
df_train.loc[(df_train['Fare'] > 170.776) & (
    df_train['Fare'] <= 314.553), 'Fare'] = 2
df_train.loc[(df_train['Fare'] > 314.553) & (
    df_train['Fare'] <= 513), 'Fare'] = 3
df_train = df_train.drop(['Fare_Band'], axis=1)


df_train['FamilySize'] = df_train['SibSp']+df_train['Parch']+1

df_train = df_train.drop(['SibSp', 'Parch'], axis=1)

df_train = df_train.drop(['Ticket'], axis=1)


df_train['Cabin'] = df_train['Cabin'].fillna('U')
df_train['Cabin'] = df_train['Cabin'].astype(str).str[0]

df_train = df_train.drop(['Cabin'], axis=1)

# get dummies
df_train = pd.get_dummies(
    columns=['Pclass', 'Sex', 'Embarked', 'Title', 'Age', 'Fare'], data=df_train)
X = df_train.drop('Survived', axis=1)
y = df_train['Survived']


# Transform and split
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=4)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler.fit(X_test)
X_test = scaler.transform(X_test)


# Classifier SelectBox
classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'Logistic Regression', 'Random Forest')
)

# Model Params


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'Logistic Regression':
        C = st.sidebar.slider('max iterations', 1, 100, 10)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 31)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 31)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params


params = add_parameter_ui(classifier_name)

# Selected Model and params


def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'Logistic Regression':
        clf = LogisticRegression(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                           max_depth=params['max_depth'], random_state=1234)
    return clf


clf = get_classifier(classifier_name, params)

#### CLASSIFICATION ####


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


st.write(f'Classifier = {classifier_name}')

# Results
st.write(f'Accuracy =', accuracy_score(y_test, y_pred).round(2))
st.write("Precision: ", precision_score(y_test, y_pred).round(2))
st.write("Recall: ", recall_score(y_test, y_pred).round(2))

#st.set_option('deprecation.showPyplotGlobalUse', False)

# Confusion Matrix
st.subheader("Confusion Matrix")
#plot_confusion_matrix(clf, X_test, y_test)
ConfusionMatrixDisplay.from_estimator(clf,X_test,y_test)
st.pyplot()
