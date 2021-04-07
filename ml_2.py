import pandas as pd

nyc = pd.read_csv("ave_hi_nyc_jan_1895-2018.csv")

# print(nyc.head(3))

# print(nyc.Date.values)

# print(nyc.Date.values.reshape(-1, 1))

# print(nyc.Temperature.values)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1, 1), nyc.Temperature.values, random_state=11
)

# print(X_train.shape)  # 93, 1
# print(X_test.shape)  # 31, 1
# print(y_train.shape)  # 93
# print(y_test.shape)  # 31


from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

# the fit method expects the samples and the targets for training
linear_regression.fit(X=X_train, y=y_train)

# print(linear_regression.coef_)  # [0.01939167]

# print(linear_regression.intercept_)  # -0.30779820252656975

predicted = linear_regression.predict(X_test)

expected = y_test

for p, e in zip(predicted[::5], expected[::5]):  # check every 5th element
    print(f"predicted: {p:.2f}, expected: {e:.2f}")

# predicted: 37.86, expected: 31.70
# predicted: 38.69, expected: 34.80
# predicted: 37.00, expected: 39.40
# predicted: 37.25, expected: 45.70
# predicted: 38.05, expected: 32.30
# predicted: 37.64, expected: 33.80
# predicted: 36.94, expected: 39.70

# lambda implements y = mx + b
predict = lambda x: linear_regression.coef_ * x + linear_regression.intercept_

print(predict(2019))  # [38.84399018]
print(predict(1890))  # [36.34246432]


import seaborn as sns

axes = sns.scatterplot(
    data=nyc,
    x="Date",
    y="Temperature",
    hue="Temperature",
    palette="winter",
    legend=False,
)

axes.set_ylim(10, 70)  # scale y-axis

import numpy as np

x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
print(x)  # [1895 2018]
y = predict(x)
print(y)  # [36.43942269 38.82459851]

import matplotlib.pyplot as plt

line = plt.plot(x, y)

plt.show()