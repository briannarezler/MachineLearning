# 1.  Create a Seaborn pairplot graph (the book has an example in Unsupervised Machine Learning
# for the Iris Dataset Section 15.7.3) for the California Housing dataset. Try the Matplotlib
# features for panning and zooming the diagram. These are accessible via the icons in the Matplotlib window.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import datasets

dataset = datasets.fetch_california_housing()

# print(dataset.data.shape)
# (20640, 8)

# print(dataset.DESCR)
#:Attribute Information:
#        - MedInc        median income in block
#        - HouseAge      median house age in block
#        - AveRooms      average number of rooms
#        - AveBedrms     average number of bedrooms
#        - Population    block population
#        - AveOccup      average house occupancy
#        - Latitude      house block latitude
#        - Longitude     house block longitude

# print(dataset.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms','AveBedrms',
#  'Population', 'AveOccup', 'Latitude', 'Longitude']

import pandas as pd

pd.set_option("max_columns", 8)

pd.set_option("display.width", None)

dataset_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

# print(dataset_df.head())
#    MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude
# 0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23
# 1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86    -122.22
# 2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85    -122.24
# 3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85    -122.25
# 4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85    -122.25

pd.set_option("precision", 2)

# print(dataset_df.describe())

import seaborn as sns

sns.set(font_scale=1.1)

sns.set_style("whitegrid")

grid = sns.pairplot(data=dataset_df, vars=dataset_df.columns[0:3])
# grid = sns.pairplot(data=dataset_df, vars=dataset_df.columns[0:8])

import matplotlib.pyplot as plt

plt.show()
