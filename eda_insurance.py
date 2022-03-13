import pandas as pd
import numpy as np
import math as mp

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as plx

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

import scipy as sp




ins= pd.read_csv('insurance.csv')
print(ins)

print(ins.shape)
print(ins.dtypes)
print(ins.columns)
print(ins.info())

# #checking null values
print(ins.isnull().sum())

print(ins.describe())

bmi_u=ins.bmi.unique()
print(bmi_u)

print(bmi_u.sort())
print(sorted(bmi_u,reverse=True))

# BMI is 18 or lower
# BMI is 18.5 to 24 -healthy
# BMI is 25.0 to 29.- overweight
# BMI is 30.0 or higher - obese range

ins_new= ins.replace(["yes","no"],[1,0])
print(ins_new)

print(ins_new.corr())

# ins_heatmap= sns.heatmap(ins_new.corr(), annot=True) 
print(ins_heatmap)

# ins_scatter_smoker_expenses= sns.scatterplot(ins["smoker"], ins["expenses"], hue=ins["smoker"])
print(ins_scatter_smoker_expenses)

# ins_scatter_bmi_expenses=sns.scatterplot(ins["expenses"],ins["bmi"], hue=ins["bmi"])
print(ins_scatter_bmi_expenses)

# ins_scatter_age_expenses=sns.scatterplot(ins["expenses"],ins["age"], hue=ins["age"])
print(ins_scatter_age_expenses)

ins_scatter_children_expenses= plt.scatter(ins["children"], ins["expenses"], color="purple")
print(ins_scatter_children_expenses)

# expenses is dependent variable and all others are independent
# heatmap showing that variables age, bmi & smoker are effecting the expenses most
# scatterplot shows people
#1. who smoke have higher expenses
#2. higher the bmi higher the expenses
#3. expenses increases as age increases
#4. scatterplot b/w children & expenses shows no correlation, hence it doesnt affect the expenses

#dataframe with age, bmi, smoker
ins_drop=ins.drop(["expenses","region","children","sex"], axis=1)
print(ins_drop)

ins_drop= ins_drop.replace(["yes","no"],[1,0])
print(ins_drop)

print(ins_drop.corr())

#LinearRegtression b/w independent variables & charges
# Equation: y = ax+b
# a = slope
# b = intercept
# x = input
# y = output
model= LinearRegression()
x= ins_drop
y= ins["expenses"]
model= LinearRegression()
model.fit(x,y)
model.score(x,y)
model.intercept_
model.coef_
y_pred= model.predict(x)
print(y_pred)

print(sns.regplot(y,y_pred,color="purple"))

print(model.fit(x,y))
print(model.score(x,y))

#intercept
print(model.intercept_)

##from model.coef_ or slopes, variable smoker affects the charges most as it has highest slope values
print(model.coef_)

#difference b/w predicted and original values
residuals= y-y_pred
print(residuals)

#bmi also affecting the charges, checking what kind of people having highest number of insurance. 

#changing to numpy series
ins_np= ins.to_numpy()
print(ins_np)

#getting count of people having different kind of health 
bmi=ins["bmi"]
bmi_np= bmi.to_numpy()
print(bmi_np)

# count of healthy people
bmi_h=bmi_np[(bmi>=18.5) & (bmi<=24)]
bmi_h_c=len(bmi_h)
print(bmi_h_c)


# count of overweight people
bmi_ow=bmi_np[(bmi>24) & (bmi<=29)]
bmi_ow_c=len(bmi_ow)
print(bmi_ow_c)


# count of obese people
bmi_ob=bmi_np[(bmi>29)]
bmi_ob_c=len(bmi_ob)
print(bmi_ob_c)


# count of weak people
bmi_w=bmi_np[(bmi<18.5)]
bmi_w_c=len(bmi_w)
print(bmi_w_c)

bmi_df_c=pd.DataFrame(data=[bmi_h_c, bmi_ow_c, bmi_ob_c, bmi_w_c],columns=["bmi"])
print(bmi_df_c)

#adding column
type=["healthy","overweight","obese","weak"]
bmi_df_c["type"]= type
print(bmi_df_c)


#number of obese people are highest, hence people with obesity having highest number of insurances 
print(sns.barplot(bmi_df_c["type"],bmi_df_c["bmi"]))



