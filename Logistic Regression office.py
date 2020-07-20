# Classification

# Logistic Regression - Linear regression of Classification
libraries
dataset
Dependent and Independent
NaN
categorical
data split
data standardization
Model Manipulation
Model Integration*
Prediction
Visualization

# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dataset
dataset = pd.read_excel ('Logistic regression Home.xlsx')

# Dependent and Independent
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,
                                                    random_state = None)

# Model Manipulation
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)
pred_prob = classifier.predict_proba(X_test)
log_prob = classifier.predict_log_proba(X_test)

# =================
# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dataset
dataset = pd.read_csv ('car churn.csv')

# Dependent and Independent
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,
                                                    random_state = None)

# data standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Model Manipulation
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, pred)
print (cm)
print (accuracy_score(y_test, pred))
[[48  4] = 0 = 48 pred are correct and 4 are incorrect (who dint buy car)
 [ 7 21]] = 1 = 21 pred are correct and 7 are incorrect (who bought car)

0.8625

