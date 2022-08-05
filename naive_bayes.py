from csv import DictReader as csv_DictReader
from collections import Counter
from math import prod, pi, sqrt, exp
from functools import partial
from random import shuffle


with open("iris.csv", newline='') as csvfile:
    reader = csv_DictReader(csvfile)
    data = list(reader)

classes = Counter(row['variety'] for row in data)
attributes = set(reader.fieldnames) - {'variety',}

shuffle(data)
prop = 0.80
num_train_samples = int(prop * len(data))
train = data[:num_train_samples]
test = data[num_train_samples:]

def calc_class_prob(x, class_value):
    subset = [row for row in data if row['variety'] == class_value]
    assert len(subset) == classes[class_value]
    print(class_value, {attribute: f"{sum(1 for row in subset if row[attribute] == x[attribute])} / {len(subset)}" for attribute in attributes})
    return prod(sum(1 for row in subset if row[attribute] == x[attribute]) / len(subset) for attribute in attributes) * (classes[class_value] / len(data))


confusion_matrix = {actual_class: {predicted_class: 0 for predicted_class in classes} for actual_class in classes}
for datapoint in test:
    actual_class = datapoint['variety']
    predicted_class = max(classes, key=partial(calc_class_prob, datapoint))
    print(f"predicted {predicted_class}, actual {actual_class}")
    confusion_matrix[actual_class][predicted_class] += 1

print()
print("confusion matrix (actual: predicted)", confusion_matrix)
num_correct = sum(confusion_matrix[class_value][class_value] for class_value in classes)
print(f"testing accuracy: {num_correct} / {len(test)}")
