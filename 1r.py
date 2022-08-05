from mlxtend.data import iris_data
from sklearn.model_selection import train_test_split
from mlxtend.classifier import OneRClassifier



X, y = iris_data()
Xd_train, Xd_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
oner = OneRClassifier()

oner.fit(Xd_train, y_train);
oner.predict(Xd_train)
test_acc = oner.score(Xd_test, y_test)
print(f'Test accuracy {test_acc*100:.2f}%')
 

values = [row for row in dataset]
prediction = max(set(values), key=values.count)
predicted = [prediction for i in range(len(dataset))]
print(predicted)