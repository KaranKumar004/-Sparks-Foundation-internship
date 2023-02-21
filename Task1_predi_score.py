# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 13:53:17 2023

@author: Deepak SK
"""

import  os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  
os.chdir('C:/Users/Deepak SK/Desktop/internship in sparks')

data=pd.read_csv('hours and scores dataset.csv')
data.columns
data
print(data.head())
sns.scatterplot(x = 'Hours', y = 'Scores', data= data)
plt.title("HOURS VS PERCENTAGE")
x = data.loc[:, ['Hours']].values
y = data.loc[:,['Scores']].values
x  
y
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(x_train,y_train)
a=regr.predict([[9.25]])
print(a)
data.columns
line = regr.coef_*x+regr.intercept_
plt.scatter(x,y)   
plt.plot(x, line)  
plt.show()
perdict=regr.predict(x_test)
perdict
a= regr.predict([[9.2]])
a
regr.score(x_test, y_test)
regr.score(x_train, y_train)
