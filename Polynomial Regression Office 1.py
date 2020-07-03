# Polynomial Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# Linear model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit (X,y)

# To pull polynomial feature
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 0)
X_poly = poly_reg.fit_transform(X)

# Integration
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualization
plt.scatter(X,y, color = 'r')
plt.plot(X, lr.predict(X))
plt.xlabel('Level')
plt.ylabel ('Salary')
plt.title ('LRM')

lr.predict([[7.5]])

# Visualization Polynomial
plt.scatter(X,y, color = 'r')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)))
plt.xlabel('Level')
plt.ylabel ('Salary')
plt.title ('Polynomial')

lin_reg_2.predict(poly_reg.fit_transform([[7.5]]))







