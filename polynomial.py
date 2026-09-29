import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
loader = pd.read_csv("FuelConsumption.csv")
observator = loader[['ENGINESIZE','CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
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
lin = linear_model.LinearRegression()
train_y_ = lin.fit(poly_train_x, train_y)
print(f"The coefficient is: {lin.coef_}")
print(f"The intercept is: {lin.intercept_}")
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color = 'blue')
plt.scatter(test.ENGINESIZE, test.CO2EMISSIONS, color = 'green')
xx = np.arange(0, 10, 0.1) 
yy = lin.intercept_[0] + lin.coef_[0][1] * xx + lin.coef_[0][2] * np.power(xx , 2)
plt.plot(xx , yy, '-r')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emission')
plt.show()
from sklearn.metrics import r2_score
poly_test_x = poly.fit_transform(test_x)
test_y_ = lin.predict(poly_test_x)
print(f"Mean Absolute Error is: {np.mean(np.absolute(test_y_ - test_y))}")
print(f"Residual Sum Of Squares is : {np.mean((test_y_ - test_y) ** 2)}")
print(f"R2_SCORE is: {r2_score(test_y , test_y_)}")

