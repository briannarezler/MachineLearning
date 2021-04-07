from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()  # bunch object

# print(california.DESCR)

# print(california.data.shape)  # (20640, 8)

# print(california.target.shape)  # (20640,)

# print(california.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
# 'Population', 'AveOccup', 'Latitude', 'Longitude']

import pandas as pd

pd.set_option("precision", 4)  # 4 digit precision for floats
pd.set_option("max_columns", 9)  # display up to 9 columns in DataFrame outputs
pd.set_option("display.width", None)  # auto-detect the display width for wrapping

# Creates the initial DataFrame using the data in california.data and with the
# column names specified based on the features of the sample
california_df = pd.DataFrame(california.data, columns=california.feature_names)

# add a column to the dataframe for the median house values stored in california.target:
california_df["MedHouseValue"] = pd.Series(california.target)

# print(california_df.head())
#   MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude  MedHouseValue
# 0  8.3252      41.0    6.9841     1.0238       322.0    2.5556     37.88    -122.23          4.526
# 1  8.3014      21.0    6.2381     0.9719      2401.0    2.1098     37.86    -122.22          3.585
# 2  7.2574      52.0    8.2881     1.0734       496.0    2.8023     37.85    -122.24          3.521
# 3  5.6431      52.0    5.8174     1.0731       558.0    2.5479     37.85    -122.25          3.413
# 4  3.8462      52.0    6.2819     1.0811       565.0    2.1815     37.85    -122.25          3.422

# The keyword argument frac specifies the fraction of the data to select (0.1 for 10%),
# and the keyword argument random_state enables you to seed the random number generator.
# this allows you to reproduce the same 'randomly' selected rows
sample_df = california_df.sample(frac=0.1, random_state=17)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=2)
sns.set_style("whitegrid")


for feature in california.feature_names:
    plt.figure(figsize=(8, 4.5))  # 8"-by-4.5" figure
    sns.scatterplot(
        data=sample_df,
        x=feature,
        y="MedHouseValue",
        hue="MedHouseValue",
        palette="cool",
        legend=False,
    )

# plt.show()


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    california.data, california.target, random_state=11
)

# print(x_train.shape) #(15480, 8)
# print(x_test.shape) #(5160, 8)
# print(y_train.shape) #(15480,)
# print(y_test.shape) #(5160,)


from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

linear_regression.fit(X=x_train, y=y_train)

"""
for i, name in enumerate(california.feature_names):
    print(f"{name:>10}: {linear_regression.coef_[i]}")
#    MedInc: 0.4377030215382204
#  HouseAge: 0.009216834565798165
#  AveRooms: -0.10732526637360909
# AveBedrms: 0.6117133073918076
# Population: -5.756822009296558e-06
#  AveOccup: -0.0033845664657163824
#  Latitude: -0.4194818609649073
# Longitude: -0.4337713349874012
"""

predicted = linear_regression.predict(x_test)
# print(predicted[:5])  # View the first 5 predictions
# [1.25396876 2.34693107 2.03794745 1.8701254  2.53608339]

expected = y_test
# print(expected[:5])  # View the first 5 expected target values
# [0.762 1.732 1.125 1.37  1.856]


# Create a DataFrame containing columns for the expected and predicted values

df = pd.DataFrame()

df["Expected"] = pd.Series(expected)

df["Predicted"] = pd.Series(predicted)

# print(df[:10])

# Plot the data as a scatter plot with the expected (target)
# prices along the x-axis and the predicted prices along the y-axis:

import matplotlib.pyplot as plt2

figure = plt2.figure(figsize=(9, 9))

axes = sns.scatterplot(
    data=df, x="Expected", y="Predicted", hue="Predicted", palette="cool", legend=False
)

# Set the x and y axes' limits to use the same scale along both axes:

start = min(expected.min(), predicted.min())
# print(start)  # -0.6830978604144633

end = max(expected.max(), predicted.max())
# print(end) # 7.155719818496827

axes.set_xlim(start, end)

axes.set_ylim(start, end)

# The following snippet displays a line between the points representing
# the lower-left corner of the graph (Start, start) and the upper-right
# corner of the graph (end, end). The third argument ('k--') indicates
# the line's style. The letter k represents the color black, and
# the -- indicates that plot should draw a dashed line:

line = plt2.plot([start, end], [start, end], "k--")

# plt2.show()
