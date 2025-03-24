"""
<h2><u>Supervised Machine Learning</u></h2>

# Module 1 – Introduction to Machine Learning
<h2> Demo 1: Data Pre-processing Techniques: Analysing Engineering Graduate Salary </h2>
"""

"""
### Dataset Attributes:
1. ID: A unique ID to identify a candidate

2. Salary: Annual CTC offered to the candidate (in INR)

3. Gender: Candidate's gender

4. 10percentage: Overall marks obtained in grade 10 examinations

5. 10board: The school board whose curriculum the candidate followed in grade 10

6. 12graduation: Year of graduation - senior year high school

7. 12percentage: Overall marks obtained in grade 12 examinations

8. 12board: The school board whose curriculum the candidate followed

9. CollegeID: Unique ID identifying the university/college which the candidate 
attended for her/his undergraduate

10. CollegeTier: Each college has been annotated as 1 or 2. The annotations have been computed from the average AMCAT scores obtained by the students in the college/university. Colleges with an average score above a threshold are tagged as 1 and others as 2.

11. Degree: Degree obtained/pursued by the candidate

12. Specialization: Specialization pursued by the candidate

13. CollegeGPA: Aggregate GPA at graduation

14. CollegeCityID: A unique ID to identify the city in which the college is located in.

15. CollegeCityTier: The tier of the city in which the college is located in. This is annotated based on the population of the cities.

16. CollegeState: Name of the state in which the college is located

17. GraduationYear: Year of graduation (Bachelor's degree)

18. English: Scores in AMCAT English section

19. Logical: Score in AMCAT Logical ability section

20. Quant: Score in AMCAT's Quantitative ability section

21. Domain: Scores in AMCAT's domain module

22. ComputerProgramming: Score in AMCAT's Computer programming section

23. ElectronicsAndSemicon: Score in AMCAT's Electronics & Semiconductor Engineering section
24. ComputerScience: Score in AMCAT's Computer Science section

25. MechanicalEngg: Score in AMCAT's Mechanical Engineering section

26. ElectricalEngg: Score in AMCAT's Electrical Engineering section

27. TelecomEngg: Score in AMCAT's Telecommunication Engineering section

28. CivilEngg: Score in AMCAT's Civil Engineering section

29. conscientiousness: Scores in one of the sections of AMCAT's personality test

30. agreeableness: Scores in one of the sections of AMCAT's personality test

31. extraversion: Scores in one of the sections of AMCAT's personality test

32. nueroticism: Scores in one of the sections of AMCAT's personality test

33. openesstoexperience: Scores in one of the sections of AMCAT's personality test

"""


#__1. How to analyse the dataset?__

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


data = pd.read_csv('1260_m1_Data Preprocessing_csv.csv') # Loading the dataste (.csv file)
print(data.head()) # Printing first 5 rows of the DataFrame

print(data.info())

#Checking missing values
print(data.isnull().sum())

"""
### Inference:

1. In the Dataset there are 2998 records of Engineering students and total of 33 columns

2. Using .info() we get a list of columns, non-null count, dtypes( Data types)

3. Using .isnull.sum - We get a list of count of values that are null in the dataset. (Here English contains 524 null values )

"""

#__2. How to differentiate between different kind of values and how to handle the null values?__

# I want to get a list of all categorical variable values
# Object will get the list of all the categorical values present in the dataset
print(data.select_dtypes('object').columns)

# Replace the missing value?
from sklearn.impute import SimpleImputer
si = SimpleImputer()
si.fit(data[['English']]) # Replace missing values with corresponding mean because
                          #underlying distribution of English scores follow a symmetrical distribution and hence, we have chosen mean.

print(data.isnull().sum())

# We cannot perform SimpleImputer method on categorical data and the dataset contains both
# So we will create a seperate dataframe with just the numerical data of the missing values and then we will replace the null values with mean

English_df = pd.DataFrame(si.transform(data[['English']].values),columns=['English'])
print(English_df.head())

#Inference - We are replacing the null value with mean for 'English' because underlying distribution of English scores follow a symmetrical distribution and hence, we have chosen mean.

data_new = data.drop(['English'],axis=1)
data_new_wo_mv = pd.concat([data_new, English_df],axis=1)
print(data_new_wo_mv.isnull().sum())

"""
### Inference:

1. In the Dataset there are 2998 records of Engineering students and total of 34 columns

2. Using .info() we get a list of columns, non-null count, dtypes( Data types)

3. Using .isnull.sum - We get a list of count of values that are null in the dataset. (Here English contains 524 null values )
"""

# Getting the list of categorical values

print("the categorical value are: ")
print(data_new_wo_mv.select_dtypes('object').columns)

# I want to create dummy variable from all of these categroical columns
# USE IT ONLY WHEN CATEGORICAL COLUMNS ARE NOMINAL
# IF ORDINAL -- USE LABELENCODER
data_numerical = pd.get_dummies(data = data_new_wo_mv,columns=['Gender', '10board', '12board', 'Specialization', 'CollegeState'],drop_first=True)

print(data_numerical.head())

#**Using Label Encoder**

print(data_numerical['Degree'].unique())

#Dropping the 'ID' variable[link text]
data_numerical.drop('ID',axis=1, inplace=True)

print(data_numerical.head())

# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'Degree'.
data_numerical['Degree']= label_encoder.fit_transform(data_new['Degree'])

print(data_numerical['Degree'].unique())

print(data_numerical.head())

data_numerical.to_csv('Engineering_graduate_salary.csv')

"""
### Inference:

1. We have used two techniques for converting categorical to numerical conversion

> - get-dummies if the category is nominal. Eg - Gender', '10board', '12board', 'Specialization', 'CollegeState

> - Label-encoder if the category is ordinal. E.g - Degree

"""

#__4. What is the correlation between the all the variables ?__

from sklearn.preprocessing import scale   #StandardScaler, MinMaxScaler

data_numerical_scaled = pd.DataFrame(scale(data_numerical),columns=data_numerical.columns)


print(data_numerical_scaled.head())

"""
# If we observe the values there's a huge imbalance in the dataset in terms of length of numbers (e.g ID is 6 digit number while salary are 8 digits or more). 
# Due to this the model will give more priority to bigger numbers that can hamper the model accuracy
#So we have to perform Scaling to compare on common grounds
"""

"""
**Standard Scaler**- The StandardScaler assumes that the data is normally distributed within each feature and will scale them such that the distribution is now centred around 0, with a standard deviation of

> Formula = xi- Mean(x)/ Stdev(x)

**Min-Max Scaler** - It  shrinks the range such that the range is now between 0 and 1 (or -1 to 1 if there are negative values).

* It works best for cases in which the standard scaler might not work so well. 
* If the distribution is not Gaussian or the standard deviation is very small, this scaler works best.

> Formula = xi-min(x) / max(x)-min(x)

**Robust Scaler**- It uses a similar method to the Min-Max scaler but it instead uses the interquartile range, rathar than the min-max, so that it is robust to outliers. 


> Formula = xi- Q1(x) / Q3(x)-Q1(x)

"""

import seaborn as sns
sns.displot(data_numerical['10percentage'])
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

#MinMax Scalar
scaler = MinMaxScaler()
df = pd.read_csv("Engineering_graduate_salary.csv")
df_minmax_scaled = df.copy()
col_names = ['10percentage', '12percentage','collegeGPA','English','Logical','Quant','Domain','ComputerProgramming','ElectronicsAndSemicon','ComputerScience','MechanicalEngg','ElectricalEngg','TelecomEngg','CivilEngg','conscientiousness','agreeableness','extraversion','nueroticism','openess_to_experience','Salary']
features = df_minmax_scaled[col_names]
df_minmax_scaled[col_names] = scaler.fit_transform(features.values)
df_minmax_scaled.to_csv('minmax.csv')

#Standard Scalar
scaler = StandardScaler()
df_standard_scaled = df.copy()
col_names = ['10percentage', '12percentage','collegeGPA','English','Logical','Quant','Domain','ComputerProgramming','ElectronicsAndSemicon','ComputerScience','MechanicalEngg','ElectricalEngg','TelecomEngg','CivilEngg','conscientiousness','agreeableness','extraversion','nueroticism','openess_to_experience','Salary']
features = df_standard_scaled[col_names]
df_standard_scaled[col_names] = scaler.fit_transform(features.values)
df_standard_scaled.to_csv('standard.csv')

#Robust Scalar
scaler = RobustScaler()
df_robust_scaled = df.copy()
col_names = ['10percentage', '12percentage','collegeGPA','English','Logical','Quant','Domain','ComputerProgramming','ElectronicsAndSemicon','ComputerScience','MechanicalEngg','ElectricalEngg','TelecomEngg','CivilEngg','conscientiousness','agreeableness','extraversion','nueroticism','openess_to_experience','Salary']
features = df_robust_scaled[col_names]
df_robust_scaled[col_names] = scaler.fit_transform(features.values)
df_robust_scaled.to_csv('robust.csv')


print("MinMax Scalar\n",df_minmax_scaled.head())
print("standard Scalar\n",df_standard_scaled.head())
print("Robust Scalar\n",df_robust_scaled.head())

df = pd.DataFrame(data[['Salary','ID','12graduation',]].values, columns=['Salary','ID','12graduation'])
print(df.head())

# MIN MAX SCALER
df_scaled = df.copy()
col_names = ['Salary','ID','12graduation']
features = df_scaled[col_names]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled[col_names] = scaler.fit_transform(features.values)
print(df_scaled.head())

# STANDARD SCALER


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df_scaled[col_names] = scaler.fit_transform(features.values)
print(df_scaled.head())

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

df_scaled[col_names] = scaler.fit_transform(features.values)
print(df_scaled.head())

"""
### Inference:

1. We have used 3 techniques for scaling. Scaling is required because it helps to compare variables on common grounds

> - get-dummies if the category is nominal. Eg - Gender', '10board', '12board', 'Specialization', 'CollegeState

> - Label-encoder if the category is ordinal. E.g - Degree

"""

#__5. Among these many numbers of variables how do we know which variable is the best correlated with the target variable ?__


import seaborn as sns
import matplotlib.pyplot as plt
#import the dataset
#df = pd.read_csv("/content/Engineering_graduate_salary.csv")
#dropping the unwanted columns
#df=df.drop(['ID','CollegeID'],axis=1)
data = data.drop(['ID','CollegeID', 'Gender','10board','12board','Specialization','CollegeState', 'Degree'],axis=1)
#getting correlations of each features in dataset
corr_mat = df.corr()
top_corr_features = corr_mat.index
#plot the figure using matplot lib
plt.figure(figsize=(20,20))
#plot heat map using seaborn
x=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()

print(data.corr())

"""
### Inference:

1. We see that percentages and CGPA are the best correlated 

2. ComputerScience, MechanicalEngg, ElectricalEngg, TelecomEngg, CivilEngg,     extraversion, nueroticism, openesstoexperience are the variables which are least correlated

"""











