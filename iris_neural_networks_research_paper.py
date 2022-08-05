from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris = datasets.load_iris()
X = iris.data
y = iris.target
scaler = MinMaxScaler()
scaler.fit(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
clf = OneVsRestClassifier(SVC()).fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("Accuracy: ", score)