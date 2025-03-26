#data analysis
import pandas as pd
import numpy as np

#dataset from sklearn
from sklearn import datasets
#machine learning
from sklearn.model_selection import train_test_split
#algorithm
from sklearn.neighbors import KNeighborsClassifier
#metrics
from sklearn.metrics import accuracy_score,confusion_matrix

iris_data=datasets.load_iris()
print("Features: ", iris_data.feature_names)
print("Labels: ", iris_data.target_names)

features=pd.DataFrame(iris_data.data)
features.columns=iris_data.feature_names
labels=pd.DataFrame(iris_data.target)
labels.columns=['class']
dataframe=pd.concat([features,labels],axis=1)
print(dataframe.head())

X_train,X_test,y_train,y_test=train_test_split(dataframe.iloc[:,0:-1],dataframe.iloc[:,-1],test_size=0.20,random_state=3)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

x1=X_train.iloc[:,0] #sepal length of X_train
x2=X_train.iloc[:,1] #sepal width of X_train
x3=X_train.iloc[:,2] #petal length of X_train
x4=X_train.iloc[:,3] #petal width of X_train

y_pred=list()
for a,b,c,d in zip( X_test.iloc[:,0], X_test.iloc[:,1], X_test.iloc[:,2], X_test.iloc[:,3]):   #a=sepal length of X_test, b=sepal width of X_test,c= petal length of X_test, d=petal width of X_train
  dist=((a-x1)**2 + (b-x2)**2 + (c-x3)**2 + (d-x4)**2)**0.5 #calculating euclidean distance
  dist=np.array(dist)
  indexes = np.argsort(dist) #sorts the values in ascending order and return their indexes
  k=3
  l2=[y_train.iloc[indexes[0]],y_train.iloc[indexes[1]],y_train.iloc[indexes[2]]]   #labels of 3 nearest instances
  y_pred.append(max(l2,key=l2.count)) #taking maximum occuring label out of 3 nearest labels

print(accuracy_score(y_test,y_pred))

model=KNeighborsClassifier(n_neighbors=3,metric='euclidean')  # here k=3
model.fit(X_train,y_train)

y_pred2=model.predict(X_test)

print(accuracy_score(y_test,y_pred2))

print(confusion_matrix(y_test,y_pred2))

