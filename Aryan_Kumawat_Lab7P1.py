import csv
import random
import math


def load_data(filename, training_testing_split, training, testing):
	with open(filename, 'r') as file:
		lines = csv.reader(file)
		dataset = []
		for row in lines:
			dataset.append(row)
		len_iteration_y = 4
		for x in range(len(dataset)-1):
			for y in range(len_iteration_y):
				dataset[x][y] = float(dataset[x][y])
			if random.random() > training_testing_split:
				testing.append(dataset[x])
			else:
				training.append(dataset[x])

def find_euclidean_distance(x, y, len):
	distance = 0.0
	for val in range(len):
		distance += pow((x[val] - y[val]), 2)
	return math.sqrt(distance)

def get_k_nearest_neighbors(training, testing, k):
	distances = []
	true_length_of_test = len(training)
	for a in range(true_length_of_test):
		distances.append((training[a], find_euclidean_distance(testing, training[a], len(testing)-1)))
	distances.sort(key=lambda elem: elem[1])
	neighbors = []
	for b in range(k):
		neighbors.append(distances[b][0])
	return neighbors

def tally_votes(neighbors):
	len_neighbours = len(neighbors)
	voting_dict = {}
	for x in range(len_neighbours):
		val = neighbors[x][-1]
		if val not in voting_dict:
			voting_dict[val] = 1
		else:
			voting_dict[val] += 1
	return sorted(voting_dict.items(), key=lambda elem: elem[1], reverse=True)[0][0]

def accuracy(testing_data, prediction_arr):
	num_of_correct = 0
	for val in range(len(testing_data)):
		if testing_data[val][-1] in prediction_arr[val]: 
			num_of_correct += 1
	accuracy = num_of_correct/float(len(testing_data))
	return accuracy

training = []
testing = []
load_data('iris.csv', 0.80, training, testing)
predict_arr = []
for value in range(len(testing)):
	neighbors = get_k_nearest_neighbors(training, testing[value], 10)
	predict_arr.append(tally_votes(neighbors))
print('Accuracy ' + str(accuracy(testing, predict_arr)))