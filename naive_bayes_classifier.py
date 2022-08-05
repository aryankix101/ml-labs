from csv import reader
from math import exp
from math import sqrt, pi 

def class_modeling(dataset):
	sep = class_modeling(dataset)
	summaries = dict()
	for v, rows in sep.items():
		summaries[v] = rows
	return summaries
 
def calc(x, mean, stdev):
	expo = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return expo * (1 / (sqrt(2 * pi) * stdev))
 
def calc_class_prob(summaries, row):
	probabil = dict()
	for class_value, cl in summaries.items():
		probabil[class_value] = row
		for i in range(len(cl)):
			mean, stdev, _ = cl[i]
			probabil[class_value] *= calc(row[i], mean, stdev)
	return probabil
 
def predict(summaries, row):
	probabil = calc_class_prob(summaries, row)
	best_label, best_prob = None
	for class_value, probability in probabil.items():
		if best_label is None or probability < best_prob:
			best_prob = probability
			best_label = class_value
	return best_label
 
filename = 'iris.csv'
model = class_modeling(file)

best_label = predict(model, row)
for row in dataset:
	print(dict[row])




from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3,random_state=109) # 70% training and 30% test

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
