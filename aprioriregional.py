

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Datasetwithservices-Sheet1.csv')
tempi = dataset.iloc[:, :].values
y = dataset.iloc[:, 12].values
X = dataset.iloc[:, 13:].values

x=[]
temp=[]

X_train=[]
y_train=[]

p=['ram','hdd','os update','office update','antivirus upgrade','warranty service']
X_f=[]
for i in range(0,len(y)):
    temp=X[i]
    if y[i] == 1 and tempi[i][8] == 'Singapore':
        for j in range(0,len(temp)):
            if temp[j] == 1:
                x.append(p[j])
            else:
                x.append('nan')
        X_f.append(x)
        X_train.append(x)
        y_train.append(1)
        x=[]
    else:
       X_f.append(['nan', 'nan', 'nan', 'nan', 'nan', 'nan']) 
    

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(X_train, min_support = 0.003, min_confidence = 0.2, min_lift = 1.0001, min_length = 2)

# Visualising the results
results = list(rules)


for i in range(0,len(results)):
    print(results[i])
    print("\n")
    
    
