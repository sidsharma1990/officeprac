# Decision Tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Importing class (Decision Tree)
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(random_state = 0)
reg.fit(X,y)

reg.predict([[6.5]])

plt.scatter(X, y)
plt.plot(X, reg.predict(X),color = 'r')
plt.title('DT')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y)
plt.plot(X_grid, reg.predict(X_grid),color = 'r')
plt.title('DT')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()













