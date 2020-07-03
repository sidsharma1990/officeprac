# Simple Linear Regression Office

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Spilitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2,
                                                    random_state = None)

# linear regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

# Prediction
reg.predict([[2.3]])
reg.predict([[11]])
y_pred = reg.predict(X_test)

reg.coef_
reg.intercept_

#### Training sets
plt.scatter(X_train, y_train)
plt.plot(X_train, reg.predict(X_train), color = 'r')
plt.xlabel('Exp')
plt.ylabel('Salary')
plt.title('SLR Training dataset')
plt.show

plt.scatter(X_test, y_test)
plt.plot(X_test, reg.predict(X_test), color = 'r')
plt.xlabel('Exp')
plt.ylabel('Salary')
plt.title('SLR Test dataset')
plt.show

######### To Save model (Pickle)
import pickle
with open('SLR Model', 'wb') as file:
    pickle.dump(reg, file)

with open('SLR Model', 'rb') as file:
    ds = pickle.load(file)

ds.predict([[11]])















