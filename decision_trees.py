from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()
X, Y = iris.data, iris.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
tree.plot_tree(clf)
