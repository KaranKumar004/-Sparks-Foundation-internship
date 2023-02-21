# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 13:53:17 2023

@author: karan SK
"""
# import the libraries
import  os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  
#change the current directary
os.chdir('C:/Users/Deepak SK/Desktop/internship in sparks')
#load the dataset as data
data=pd.read_csv('hours and scores dataset.csv')
data.columns
data
#use head function to see the first 5 rows in the dat
print(data.head())
#visualize data to see if there is any patterns in it 
sns.scatterplot(x = 'Hours', y = 'Scores', data= data)
plt.title("HOURS VS PERCENTAGE")
#separate the dependent and independent variables
x = data.loc[:, ['Hours']].values
y = data.loc[:,['Scores']].values
x  
y
#split the data as train and test sets
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#import the model
from sklearn.linear_model import LinearRegression
regr = LinearRegression()

#fit the data into the model
regr.fit(x_train,y_train)
a=regr.predict([[9.25]])
print(a)
#visualize the Regression line
line = regr.coef_*x+regr.intercept_
plt.scatter(x,y)   
plt.plot(x, line)  
plt.show()
#perdict for the test data
perdict=regr.predict(x_test)
perdict
#perdict the score for 9.2 hours of study
a= regr.predict([[9.2]])
a
#accuracy check
regr.score(x_test, y_test)
regr.score(x_train, y_train)
