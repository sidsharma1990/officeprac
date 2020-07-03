# Polynomial Regression 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_excel('Age polynomial regression.xlsx')

y = dataset.iloc[:,-1].values
X = dataset.iloc[:,:-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Linear model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit (X_train, y_train)

# To pull polynomial feature
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures()
X_poly = poly_reg.fit_transform(X_train)

# Integration
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)

x
# Visualization Polynomial
plt.scatter(X_train, y_train, color = 'r')
plt.plot(X_train, lin_reg_2.predict(poly_reg.fit_transform(X_train)))
plt.xlabel('Level')
plt.ylabel ('Salary')
plt.title ('Polynomial')

lin_reg_2.predict(poly_reg.fit_transform([[23]]))
lin_reg_2.predict(poly_reg.fit_transform([[17]]))
lin_reg_2.predict(poly_reg.fit_transform([[17.5]]))
lin_reg_2.predict(poly_reg.fit_transform([[18]]))
lin_reg_2.predict(poly_reg.fit_transform([[18.5]]))
# pred = lin_reg_2.predict(poly_reg.fit_transform([X_test]))


















