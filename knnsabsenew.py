# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('datasetforknn.csv')
X_test= [{"Year":2009,"RAM":2,"HDD":180,"Location":'USA',"Warranty":3,"Age":40}]

x_t=[[X_test[0]['Year'], X_test[0]['RAM'],X_test[0]['HDD'],X_test[0]['Location'], X_test[0]['Warranty'], X_test[0]['Age']]]

X = dataset.iloc[:, 3:9].values
xtemp=np.ndarray.tolist(X)
xtemp=xtemp+x_t
xtemp=np.array(xtemp)
X=xtemp


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

E=X[:-1,:]
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(E, y)


X_yo=X[2000]
# Predicting the Test set results

X_yo=X_yo.reshape(1, -1)
y_pred = classifier.predict(X_yo)

