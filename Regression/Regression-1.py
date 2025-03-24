#Importing the required libraries we need to:
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score , mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures , LabelEncoder

data=pd.read_csv("1260_m2_All_Regression_csv.csv")
print(data.head())

print(data.describe())

print(data.region.unique())

print(data.isnull().sum())

hist = data.hist(bins=100,color='red',figsize=(8, 8))
plt.show()
# Visulization of Data is done when it's required it's not a thumb rule to do so everytime

data.info()

le = LabelEncoder()
data["sex"] = le.fit_transform(data["sex"])
print(data["sex"].unique())

data["smoker"] = le.fit_transform(data["smoker"])
print(data["smoker"].unique())

data["region"] = le.fit_transform(data["region"])
print(data["region"].unique())

print(data.head())

hist = data.hist(bins=100,color='red',figsize=(8, 8))
plt.show()

sns.heatmap(data.corr(),annot=True);
plt.show()

# In supervised learning we tell the model what is my Dependent Variable and what is my Independent Variable
# Here in this case my dependent variable is charges and Independent variable is age
X = data[['age']]
Y = data['charges']

print(X)
print(Y)

# Called Linear Regression Class to perform Linear Regression
from sklearn.linear_model import LinearRegression

# Creating an empty object of Linear Regression class
lg_model = LinearRegression()

lg_model.fit(X,Y)

print(lg_model.coef_)   # m = slope
print(lg_model.intercept_) # c = intercept

from sklearn.metrics import r2_score
print(r2_score(Y,lg_model.predict(X)))

# For every given X value using the above equation, what is the predicted Y value
Y_pred = lg_model.predict(X)
print(Y_pred)

plt.scatter(data['age'],data['charges'])
plt.plot(data['age'],Y_pred,c = 'red')
plt.show()


# In supervised learning we tell the model what is my Dependent variable and what is my Independent variable
X = data[['age']]
Y = data['charges']

# We will have to divide the data into two parts, one for training the model and other for testing the model
from sklearn.model_selection import train_test_split

# we choose spit percentage based on the project objectives and training set, testing set repersentativess, common split is 80-20, 70-30

# test_Size = 30% of the total data will be part of test datasets
# train_test_split does a random selection of subset
# every time we run this part of the code - we will have differet datapoints as part of train & test
# Because every time because train & test data will change the m and c value will change
# and the error will change
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 42)

# Running linear Regression on train data
lm_model = LinearRegression()
lm_model.fit(X_train,Y_train)
print(lm_model.coef_)
print(lm_model.intercept_)

from sklearn.metrics import r2_score
print(r2_score(Y_train,lg_model.predict(X_train)))

y_pred = lm_model.predict(X_test)

df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
df1 = df.head()
print(df1)

from sklearn.metrics import r2_score
print(r2_score(Y_test,lm_model.predict(X_test)))



import statsmodels.api as sm

X = data['age']
y = data['charges']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

sns.heatmap(data.corr(),annot=True)
plt.show()

# Building a Multiple Linear Regression
# Understanding the data so that 2 dependent varaibles can be taken for the model as input

data.head()

# Multiple linear regression ( we will be having more independent variables)
lr_data = data[['age','smoker','charges']]
X = lr_data[['age','smoker']]
Y = lr_data['charges']

#checking the magnitude of coefficients

predictors = X_train.columns
coef = pd.Series(lg_model.coef_,predictors).sort_values()
coef.plot(kind='bar', title='Modal Coefficients')
plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
#ax = Axes3D(fig)
ax = fig.add_subplot(projection='3d')
ax.scatter(lr_data['age'],lr_data['smoker'],lr_data['charges'])
ax.set_xlabel("age")
ax.set_ylabel("smoker")
ax.set_zlabel("charges")
plt.show()

# We will have to divide the data into two parts
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 42)


## Run linear Regression on train data
lm_model = LinearRegression()
lm_model.fit(X_train,Y_train)
print(lm_model.coef_)
print(lm_model.intercept_)

from sklearn.metrics import r2_score
print(r2_score(Y,lm_model.predict(X)))

print(X_train.columns)

print(X_train['age'])
Y_train_ser = pd.Series(Y_train.values)
print(Y_train_ser.shape)

Y_pred_train = pd.Series(lm_model.predict(X_train))
print(Y_pred_train.shape)

fig = plt.figure()
#ax = Axes3D(fig)
ax = fig.add_subplot(projection='3d')
ax.scatter(X_train['age'],X_train['smoker'],Y_train)
ax.scatter(X_train['age'],X_train['smoker'],Y_pred_train,color = 'red')
ax.set_xlabel("age")
ax.set_ylabel("smoker")
ax.set_zlabel("charges")
plt.show()

Y_pred_test = lm_model.predict(X_test)
#Y_pred_test

# LINEAR REGRESSION MODEL WIRH 2 IDVs
r_Square2= r2_score(Y_train,lm_model.predict(X_train))
print(r_Square2)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

n_samples = 100
X = np.linspace(0, 10, 100)
y = X ** 4 + np.random.randn(n_samples) * 100 + 100
plt.figure(figsize=(10,8))
plt.scatter(X, y)
plt.show()

lin_reg = LinearRegression()
lin_reg.fit(X.reshape(-1, 1), y)
model_pred = lin_reg.predict(X.reshape(-1,1))
plt.figure(figsize=(10,8));
plt.scatter(X, y);
plt.plot(X, model_pred)
plt.show()
print(r2_score(y, model_pred))

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X.reshape(-1, 1))

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y.reshape(-1, 1))
y_pred = lin_reg_2.predict(X_poly)
plt.figure(figsize=(10,8))
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.show()
print(r2_score(y, y_pred))














