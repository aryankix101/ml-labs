from math import exp
from math import sqrt, pi 
from csv import DictReader as csv_DictReader
from collections import Counter


with open("iris.csv", newline='') as csvfile:
        reader = csv_DictReader(csvfile)
    data = list(reader)
classes = Counter(row['variety'] for row in data)

def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated



def class_modeling(dataset):
	sep = separate_by_class(dataset)
	summaries = dict()
	for v, rows in sep.items():
		summaries[v] = rows
	return summaries
 
def calc(x, mean, stdev):
	expo = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return expo * (1 / (sqrt(2 * pi) * stdev))
 
def predict(summaries, row):
	probabil = calc_class_prob(summaries, row)
	best_label= None
	for class_value, probability in probabil.items():
		if best_label is None:
			best_label = class_value
	return best_label

def calc_class_prob(summaries, row):
	probabil = dict()
	for class_value, cl in summaries.items():
		probabil[class_value] = row
		for i in range(len(cl)):
			mean, stdev, _ = cl[i]
			probabil[class_value] *= calc(row[i], mean, stdev)
	return probabil

dataset = data
model = summarize_by_class(dataset)
best_label = predict(model, row)
for row in dataset:
	print(dict[row])
