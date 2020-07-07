# SVR - Support Vector Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y),1)

dataset.corr()

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Importing class
from sklearn.svm import SVR
reg = SVR()
reg.fit(X,y)

reg.predict([[6.5]])

reg.predict(sc_X.transform([[6.5]]))

sc_y.inverse_transform(reg.predict(sc_X.transform([[6.5]])))

# Visualization
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y))
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(reg.predict(X)),
         color = 'r')
plt.title('SVR')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()



















