# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('datasetforknn.csv')


X = dataset.iloc[:, 3:9].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
X[:,1]=labelencoder_X.fit_transform(X[:,1])
X[:,2]=labelencoder_X.fit_transform(X[:,2])
X[:,3]=labelencoder_X.fit_transform(X[:,3])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

onehotencoder = OneHotEncoder(categorical_features=[9])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

onehotencoder = OneHotEncoder(categorical_features=[13])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

onehotencoder = OneHotEncoder(categorical_features=[18])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


y=dataset.iloc[:,12].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)


# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X, y)


X_test=X[1999]
# Predicting the Test set results

X_test=X_test.reshape(1, -1)
y_pred = classifier.predict(X_test)

