# The Iris dataset is referred to as a “toy dataset” because it has only 150 samples and four features.
# The dataset describes 50 samples for each of three Iris flower species—Iris setosa, Iris versicolor and Iris
# virginica. Each sample’s features are the sepal length, sepal width, petal
# length and petal width, all measured in centimeters. The sepals are the larger outer parts of each flower
# that protect the smaller inside petals before the flower buds bloom.


# EXERCISE
# load the iris dataset and use classification
# to see if the expected and predicted species
# match up
from sklearn.datasets import load_iris

iris = load_iris()

from sklearn.model_selection import train_test_split


data_train, data_test, target_train, target_test = train_test_split(
    iris.data, iris.target, random_state=11
)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X=data_train, y=target_train)


# display the shape of the data, target and target_names

print(iris.data.shape)  # (150, 4)

print(iris.target.shape)  # (150, )

print(iris.target_names.shape)  # (3, )


# display the first 10 predicted and expected results using
# the species names not the number (using target_names)

predicted = knn.predict(X=data_test)

expected = target_test

print(predicted[:10])  # [2 2 2 1 2 0 1 0 0 1]
print(expected[:10])  # [2 2 2 1 2 0 1 0 0 1]

print([iris.target_names[x] for x in predicted[:10]])
print([iris.target_names[x] for x in expected[:10]])


# display the values that the model got wrong

wrong = [(p, e) for (p, e) in zip(predicted, expected) if p != e]

print(wrong)  # [(1, 2)]

print(format(knn.score(data_test, target_test), ".2%"))  # 97.37%


# visualize the data using the confusion matrix

from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_true=expected, y_pred=predicted)

print(confusion)

"""
[[12  0  0]
 [ 0 11  0]
 [ 0  1 14]]
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt2

confusion_df = pd.DataFrame(confusion, index=range(3), columns=range(3))

figure = plt2.figure(figsize=(7, 6))
axes = sns.heatmap(confusion_df, annot=True, cmap=plt2.cm.nipy_spectral_r)
plt2.show()

print("Done")