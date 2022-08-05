import numpy as np
import math
import random
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

def check_point(point):
    t = point[0]**2+point[1]**2
    s = math.sqrt(t)
    if s<1:
        return True
    else:
        return False
 
def step(num):
    if num>0:
        return 1
    else:
        return 0
 
def sigmoid(num):
    return 1/(1+(math.e**(-num)))
 
def perceptron_network(A, x, weight_list, bias_list):
    new_f = np.vectorize(A)  # This creates a function that applies the original function to each element of a matrix individually
    a_0 = x
    a_l_prev = a_0
    for w, b in zip(weight_list[1:], bias_list[1:]):
        a_L = new_f(a_l_prev@w+b)
        a_l_prev = a_L
    return a_L
 
weight_list = [None, np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]]), np.array([[1], [1], [1], [1]])]
bias_list = [None, np.array([1, 1, 1, 1]), np.array([-2.78])]
points = []
learning_rate = None
"""for i in range(0, 500):
    temp = []
    for x in range(0, 2):
        temp.append(random.uniform(0, 1))
    points.append(tuple(temp))"""
points_correct = 0
points_incorrect = 0
print("Incorrectly classified coordinates: ")
for point in X:
    output_circle = perceptron_network(sigmoid, point, weight_list, bias_list)
    o = round(output_circle[0])            
    if (check_point(point)==False and o==0) or (check_point(point)==True and o==1):
        points_correct+=1
    else:
        points_incorrect+=1
        print(point)
print("Total correctly classified points: " + str(points_correct))
print("Total incorrectly classified points: " + str(points_incorrect))
decimal_correct = (points_correct/len(X))
percentage = "{:.0%}".format(decimal_correct)
print("Classified correctly percentage: " + str(percentage))

