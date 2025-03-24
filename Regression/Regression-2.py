import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]


"""

:Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of black people by town
        - LSTAT    % lower status of the population

"""
cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']


boston_df = pd.DataFrame(data, columns=cols)
print(boston_df.head(2))
print(boston_df.describe())

boston_df['Houseprice']=target
print(boston_df.head())

real_x=boston_df.iloc[:,0:13].values
real_y=boston_df.iloc[:,13].values

training_x,test_x,training_y,test_y= train_test_split(real_x,real_y,test_size=0.3, random_state=0)

scaler=StandardScaler()
training_x=scaler.fit_transform(training_x)
test_x=scaler.fit_transform(test_x)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

lm=LinearRegression()
lasso=Lasso()
ridge=Ridge()
elastic=ElasticNet()

lm.fit(training_x,training_y)
lasso.fit(training_x,training_y)
ridge.fit(training_x,training_y)
elastic.fit(training_x,training_y)

plt.figure(figsize=(15,10))
important_coeff=pd.Series(lm.coef_, index=cols)
important_coeff.plot(kind="barh")
plt.show()

plt.figure(figsize=(15,10))
important_coeff=pd.Series(ridge.coef_, index=cols)
important_coeff.plot(kind="barh")
plt.show()

plt.figure(figsize=(15,10))
important_coeff=pd.Series(lasso.coef_, index=cols)
important_coeff.plot(kind="barh")
plt.show()

plt.figure(figsize=(15,10))
important_coeff=pd.Series(elastic.coef_, index=cols)
important_coeff.plot(kind="barh")
plt.show()

pred_test_lm= lm.predict(test_x)
pred_test_ridge=ridge.predict(test_x)
pred_test_lasso=lasso.predict(test_x)
pred_test_elastic=elastic.predict(test_x)

print("Simple linear regression mean square error for  test data is")
print(np.round(metrics.mean_squared_error(test_y,pred_test_lm),2))

print("Ridge regression mean square error for  test data is")
print(np.round(metrics.mean_squared_error(test_y,pred_test_ridge),2))

print("Lasso regression mean square error for  test data is")
print(np.round(metrics.mean_squared_error(test_y,pred_test_lasso),2))

print("Elastic net regression mean square error for  test data is")
print(np.round(metrics.mean_squared_error(test_y,pred_test_elastic),2))

print('Rsquare value for simple regression on test data is')
print(np.round(lm.score(test_x,test_y)*100,2))

print('Rsquare value for ridge regression on test data is')
print(np.round(ridge.score(test_x,test_y)*100,2))

print('Rsquare value for lasso regression on test data is')
print(np.round(lasso.score(test_x,test_y)*100,2))

print('Rsquare value for elastic regression on test data is')
print(np.round(elastic.score(test_x,test_y)*100,2))

from sklearn.model_selection import GridSearchCV

#for ridge regression
parameters={'alpha':[1,5,10,20,30,35,40,45,50,55,100]}

ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(training_x,training_y)
print(ridge_regressor.best_params_)

#best score = Mse

print(ridge_regressor.best_score_)

#for lasso regression
parameters={'alpha':[1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}

lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(training_x,training_y)
print(lasso_regressor.best_params_)

print(lasso_regressor.best_score_)


#for elastic regression
parameters={'alpha':[1e-3,1e-2,1e-4,1,5,10,20,30,35,40,45,50,55,100]}

elastic_regressor=GridSearchCV(elastic,parameters,scoring='neg_mean_squared_error',cv=5)
elastic_regressor.fit(training_x,training_y)
print(elastic_regressor.best_params_)

print(elastic_regressor.best_score_)

import statsmodels.api as sm

real_x = sm.add_constant(real_x)  # adding a constant

model = sm.OLS(real_y, real_x).fit()
predictions = model.predict(real_x)

print_model = model.summary()
print(print_model)








