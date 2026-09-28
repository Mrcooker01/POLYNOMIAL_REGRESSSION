import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
loader = pd.read_csv("FuelConsumption.csv")
observator = loader[['ENGINESIZE','CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
plt.scatter(observator.ENGINESIZE, observator.CO2EMISSIONS, color = 'blue')
plt.xlabel('Engine Size')
plt.ylabel('Emission')
plt.show()
msk = np.random.rand(len(loader)) < 0.8
train = observator[msk]
test = observator[~msk]
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
poly = PolynomialFeatures(degree = 2)
poly_train_x = poly.fit_transform(train_x)
print(poly_train_x)
