# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib as pyplot
import pandas as pd

#loading the dataset
dataset=pd.read_csv("Data.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

## Taking care of missing data
from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

imputer = imputer.fit(X[:, 1:3])  ## Upperbound is excluded in python)
X[:, 1:3] = imputer.transform(X[:, 1:3])

## Taking care of categorical variables
