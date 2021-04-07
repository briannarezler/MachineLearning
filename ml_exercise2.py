# 2. Re-implement the simple linear regression case study of Section 15.4 using the average yearly
# temperature data. How does the temperature trend compare to the average January high temperatures?

import pandas as pd

nyc = pd.read_csv("ave_yearly_temp_nyc_1895-2017.csv")

print(nyc.head(3))
#     Date  Value  Anomaly
# 0  189512   52.1     -1.8
# 1  189612   52.3     -1.6
# 2  189712   52.3     -1.6

# print(nyc.Date.values)

# print(nyc.Date.values.reshape(-1, 1))

# print(nyc.Value.values)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1, 1), nyc.Value.values, random_state=11
)

# print(X_train.shape)  # 92, 1
# print(X_test.shape)  # 31, 1
# print(y_train.shape)  # 92
# print(y_test.shape)  # 31


from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

# the fit method expects the samples and the targets for training
linear_regression.fit(X=X_train, y=y_train)

print(linear_regression.coef_)  # [0.00031574]

print(linear_regression.intercept_)  # -7.896723314283953

predicted = linear_regression.predict(X_test)

expected = y_test


for p, e in zip(predicted[::5], expected[::5]):  # check every 5th element
    print(f"predicted: {p:.2f}, expected: {e:.2f}")

# predicted: 54.21, expected: 52.40
# predicted: 54.31, expected: 53.70
# predicted: 52.86, expected: 51.40
# predicted: 53.27, expected: 53.90
# predicted: 55.48, expected: 55.40
# predicted: 53.90, expected: 54.90
# predicted: 52.76, expected: 54.60


# lambda implements y = mx + b
predict = lambda x: linear_regression.coef_ * x + linear_regression.intercept_

import seaborn as sns

axes = sns.scatterplot(
    data=nyc,
    x="Date",
    y="Value",
    hue="Value",
    palette="winter",
    legend=False,
)

axes.set_ylim(45, 65)  # scale y-axis

import numpy as np

x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
print(x)  # [189512 201712]
y = predict(x)
print(y)  # [51.94061915 55.79270017]

import matplotlib.pyplot as plt

line = plt.plot(x, y)

plt.show()
