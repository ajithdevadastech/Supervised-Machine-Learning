import warnings
print(warnings.filterwarnings('always'))
print(warnings.filterwarnings('ignore'))


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

data = pd.read_csv("1260_m3_Evaluating_Regression_models_csv.csv")

print(data.shape)

#to find number of nulls on each column.
print(data.isnull().sum())

#to find % of nulls on each column.
print(data.isnull().sum()/data.shape[0] *100)

print(data.info())

categorical = data.select_dtypes(include =[object])
print("Categorical Features in data Set:",categorical.shape[1])

numerical= data.select_dtypes(include =[np.float64,np.int64])
print("Numerical Features in data Set:",numerical.shape[1])

print(data.describe())

print(data.columns)

print(data['Item_Weight'].isnull().sum())

plt.figure(figsize=(8,5))
#sns.boxplot('Item_Weight',data=data)
sns.boxplot(data=data['Item_Weight'])
plt.show()

data['Item_Weight']= data['Item_Weight'].fillna(data['Item_Weight'].mean())

print(data['Item_Weight'].isnull().sum())

print(data['Outlet_Size'].isnull().sum())

print(data['Outlet_Size'].value_counts())

data['Outlet_Size']= data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0])

print(data['Outlet_Size'].isnull().sum())

print(data.columns)

print(data.head())

print(data['Item_Fat_Content'].value_counts())

data['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'],inplace = True)

data['Item_Fat_Content']= data['Item_Fat_Content'].astype(str)

print(data['Item_Fat_Content'].value_counts())

data['Years_Established'] = data['Outlet_Establishment_Year'].apply(lambda x: 2020 - x)

print(data.head())

plt.figure(figsize=(8,5))
sns.countplot(data=data['Item_Fat_Content'],palette='ocean')
plt.show()

plt.figure(figsize=(25,7))
sns.countplot(data=data['Item_Type'],palette='spring')
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(data=data['Outlet_Size'],palette='summer')
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(data=data['Outlet_Location_Type'],palette='autumn')
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(data=data['Outlet_Type'],palette='twilight')
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(data=data['Years_Established'],palette='mako')
plt.show()

print(data.head())

le = LabelEncoder()
x = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type']

for i in x:
    data[i] = le.fit_transform(data[i])

print(data.head())

data = data.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year'],axis=1)

print(data.columns)

X= data[['Item_Weight','Item_Fat_Content','Item_Visibility','Item_Type','Item_MRP','Outlet_Size','Outlet_Location_Type','Outlet_Type','Years_Established']]
y= data['Item_Outlet_Sales']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=22)

features= ['Item_Weight','Item_Fat_Content','Item_Visibility','Item_Type','Item_MRP','Outlet_Size','Outlet_Location_Type','Outlet_Type','Years_Established']

LR = LinearRegression()
LR.fit(X_train,y_train)
y_pred = LR.predict(X_test)
coef2 = pd.Series(LR.coef_,features).sort_values()

print(y_pred)

print(coef2)

plt.figure(figsize=(8,5))
sns.barplot(LR.coef_)
plt.show()

y_pred = LR.predict(X_test)
##Comparing the actual output values with the predicted values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df.head())

from sklearn.metrics import r2_score
R2 = r2_score(y_test,y_pred)
print(R2)

MAE= metrics.mean_absolute_error(y_test,y_pred)
MSE= metrics.mean_squared_error(y_test,y_pred)

print("mean absolute error:",MAE)
print("mean squared error:",MSE)

from math import sqrt
rmse = sqrt(MSE)
print("Root Mean Squared Error:",rmse)












