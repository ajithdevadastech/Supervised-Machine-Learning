  #Importing Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
import ydata_profiling
from pandas_profiling import ProfileReport
import plotly.express as px

#Loading the dataset

df = pd.read_csv('employee_data.csv')

print(df.head()) #Printing the first 5 rows of dataframe


prof = ProfileReport(df)
prof.to_file(output_file='output.html') #Generating a Data Report

ProfileReport(df).to_notebook_iframe()


fig = px.histogram(df, x ="average_montly_hours")
fig.show()

fig = px.histogram(df, x = 'satisfaction_level')
#fig.show()

plt.figure(figsize=(12,8))

ax = sns.countplot(df["quit"], color='green')
for p in ax.patches:
    x = p.get_bbox().get_points()[:,0]

    y = p.get_bbox().get_points()[1,1]

    ax.annotate('{:.2g}%'.format(100.*y/len(df)), (x.mean(), y), ha='center', va='bottom')
plt.savefig("1.png")

plt.figure(figsize=(12,8))

sns.countplot(data=df,x=df['department'],hue="quit")

plt.xlabel('Departments')
plt.ylabel('Frequency')

plt.savefig("2.png")

df_new = pd.crosstab(df['salary'], df['quit'])

df_new.plot(kind = 'bar')

plt.title('Employee Attrition Frequency based on Salary')
plt.xlabel('Salary')
plt.ylabel('Frequency')

plt.savefig("3.png")

px.scatter(df, x=df['satisfaction_level'],y=df['time_spend_company'],color=df['quit'])

fig = px.box(df, x="department",y="number_project")
fig.show()

X = df.drop('quit', axis = 1)
y = df.quit

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size=0.2, stratify = y)

cat_vars = ['department', 'salary']

for vars in cat_vars:
  cat_list = pd.get_dummies(X_train[vars], prefix=vars)
  X_train = X_train.join(cat_list)

for vars in cat_vars:
  cat_list = pd.get_dummies(X_test[vars], prefix=vars)
  X_test = X_test.join(cat_list)

#Let us drop the department and salary columns

X_train.drop(columns=['department', 'salary'], axis = 1, inplace=True)
X_train.shape

X_test.drop(columns=['department', 'salary'], axis = 1, inplace=True)
X_test.shape

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import export_graphviz # display the tree within a Jupyter notebook
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from ipywidgets import interactive, IntSlider, FloatSlider, interact
import ipywidgets
from IPython.display import Image, HTML
from subprocess import call
import matplotlib.image as mpimg

@interact #To convert any function into an inteactive one just write "@interact" immediately before the function definition

def plot_tree(
    crit = ['gini', 'entropy'],
    split = ['best','random'],
    depth = IntSlider(min = 1, max = 25, value =2, continuous_update = False),
    min_split = IntSlider(min = 1, max = 5, value =2, continuous_update = False),
    #min_split is the minimum number of samples  required to split an internal node in our decision tree
    min_leaf = IntSlider(min = 1, max = 5, value =1, continuous_update = False)):

  estimator = DecisionTreeClassifier(criterion=crit,
                                     splitter=split,
                                     max_depth = depth,
                                     min_samples_split = min_split,
                                     min_samples_leaf = min_leaf
                                     )
  estimator.fit(X_train, y_train)
  print('Decision Tree Training Accuracy:', accuracy_score(y_train, estimator.predict(X_train)))
  print('Decision Tree Testing Accuracy:', accuracy_score(y_test, estimator.predict(X_test)))

  a = accuracy_score(y_train, estimator.predict(X_train))
  b = accuracy_score(y_test, estimator.predict(X_test))

  if a > 0.99:
    print('Decision Tree Training Accuracy',a, 'Decision Tree Testing Accuracy', b)
    print('Criterion:',crit,'\n', 'Split:', split,'\n', 'Depth:', depth,'\n', 'Min_split:', min_split,'\n', 'Min_leaf:', min_leaf,'\n')

  #Let us use GraphViz to export the model and display it as an image on the screen
  graph = Source(tree.export_graphviz(estimator, out_file='tree.dot',
                                      feature_names = X_train.columns,
                                      class_names = ['stayed', 'quit'],
                                      filled = True))
  display(Image(data=graph.pipe(format = 'png')))

@interact

def plot_tree_rf(crit=['gini', 'entropy'],
                depth=IntSlider(min=1, max=20, value=3, continuous_update=False),
                forests=IntSlider(min=1, max=1000, value=100, continuous_update=False),
                min_split=IntSlider(min=2, max=5, value=2, continuous_update=False),
                min_leaf=IntSlider(min=1, max=5, value=1, continuous_update=False)):

    estimator = RandomForestClassifier(
          random_state=1,
          criterion=crit,
          n_estimators=forests,
          max_depth=depth,
          min_samples_split=min_split,
          min_samples_leaf=min_leaf,
          n_jobs=-1,
          verbose=False)

    estimator.fit(X_train, y_train)

    print('Random Forest Training Accuracy:', accuracy_score(y_train, estimator.predict(X_train)))
    print('Random Forest Testing Accuracy:', accuracy_score(y_test, estimator.predict(X_test)))

    a = accuracy_score(y_train, estimator.predict(X_train))
    b = accuracy_score(y_test, estimator.predict(X_test))

    if a > 0.99:
        print('Random Forest Training Accuracy', a, 'Random Forest Testing Accuracy', b)
        print('Criterion:', crit, '\n', 'Depth:', depth, '\n', 'forests:', forests, '\n', 'Min_split:', min_split,
            '\n', 'Min_leaf:', min_leaf, '\n')


import pycaret.classification as pc

#Loading the dataset
import pandas as pd
df = pd.read_csv('employee_data.csv')

print(df.head()) #Printing the first 5 rows of dataframe

print(df['department'].unique())

pc.setup(df, target='quit')

pc.compare_models()

rf_model = pc.create_model('rf') #Performs K-Fold (10) CV for the selected model

tuned_rf = pc.tune_model(rf_model)

print(rf_model)

print(tuned_rf)

tuned_rf_eval = pc.evaluate_model(tuned_rf)

print(tuned_rf_eval)









