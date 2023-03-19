import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import math
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import statistics

df = pd.read_csv("F:\\Setups\\py-master\\py-master\\ML\\9_decision_tree\\Exercise\\titanic.csv")


# print(df.head())

def age_mean():
    mean = math.floor(df["Age"].mean())
    df["Age"] = df["Age"].fillna(mean)


def age_mode():
    mode = math.floor(df['Age'].mode())
    df['Age'] = df['Age'].fillna(mode)


def age_median():
    median = math.floor(df['Age'].median())
    df['Age'] = df['Age'].fillna(median)


X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

ohe = ColumnTransformer([('one_hot_encoding', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ohe.fit_transform(X), dtype='float')  # male = 0 1 and female = 1 0
X = X[:, 1:]

# print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

reg = DecisionTreeClassifier()
reg.fit(X_train, y_train)
print(reg.score(X_test, y_test))

'''
1 = Survived
2 = Didn't survive
'''
age_mean()
# age_mode()
# age_median()
print(reg.predict([[1, 3, 20, 8.05]]))  # Doesn't survive but we getting survived
print(reg.predict([[0, 1, 38, 71.2833]]))  # Survived and we getting survived
