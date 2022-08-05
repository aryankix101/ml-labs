from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import numpy as np
import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
clf = Perceptron(tol=1e-4, random_state=0)
clf.fit(X, y)
print("Accuracy: " + str(clf.score(X, y)))
