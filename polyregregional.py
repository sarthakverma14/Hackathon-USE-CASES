
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Datasetwithservices-Sheet1.csv')

temp = dataset.iloc[:, :].values


laptop_count=[]
year=[]
for i in range(2009,2019):
    year.append(i)


tempc=0
for j in range(0,len(year)):
    for i in range(0,len(temp)):
        if temp[i][3]=='Laptop' and temp[i][8]=='USA':
            if temp[i][4] == year[j]:
                tempc=tempc+1       
    laptop_count.append(tempc)
    tempc=0    
    
t1=[]
y=[]
for j in range(0,len(year)):
    t=year[j]
    t1.append(t)
    y.append(t1)
    t1=[]
X=y
y=laptop_count
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(year), max(year), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(2019))

