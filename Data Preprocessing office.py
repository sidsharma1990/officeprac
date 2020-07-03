# Data Preprocessing
# X is also called matrix of features

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_excel('Data-Preprocessing.xlsx')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# y1 = dataset.loc[:, 'Purchased']
# X1 = dataset.iloc[:, :-1]

###### Filling Nan Valuev in Pandas
# dataset['Age'].fillna(value = dataset['Age'].median(), inplace = True)
# dataset.Age = dataset.Age.fillna(dataset['Age'].median())
# df2 = dataset.fillna({'Age': dataset['Age'].mean(),
#                       'Salary':dataset['Salary'].median()})

# Through Scikit learn
# Imputer
# fit and transform (sepratly)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# fit and transform together
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

############ Correlation
dataset[['Age', 'Salary']].corr()

# Dummy Variable
# dummy= pd.get_dummies(dataset['State'])
# X1 = pd.concat([X1, dummy], axis = 1)
# X1.drop(['State', 'Mumbai'], axis = 1, inplace = True)

# Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), 
                                        [0])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

# for dependent variable y
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
 
# Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2)
print (X_train)
print (y_train)
print (X_test, y_test)

# Feature Scaling (Standardization) (-3 to +3)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
X_test[:,3:] = sc.transform(X_test[:,3:])

# normalizer (-2 to +2)
from sklearn.preprocessing import Normalizer
normal = Normalizer()
X_train[:,3:] = normal.fit_transform(X_train[:,3:])
X_test[:,3:] = normal.transform(X_test[:,3:])









