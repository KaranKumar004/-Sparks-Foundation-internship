# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 17:42:49 2023

@author: karan SK
"""
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


os.chdir('C:/Users/Deepak SK/Desktop/internship in sparks')
iris = pd.read_csv('Iris (1).csv')
iris.head(5)
iris.data
iris.target
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

plt.figure(figsize=(15,10))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
