import numpy as np


def perceptron(w, b, x):
    v = np.dot(w, x) + b
    y = step(v)
    return y

def step(num):
    if num>0:
        return 1
    else:
        return 0
 
def or_function(x):
    b = -0.5
    w = np.array([1, 1])
    return perceptron(w, b, x)

def not_function(x):
    b = 0.5
    w = -1
    return perceptron(w, b, x)
  
def and_function(x):
    b = -1.5
    w = np.array([1, 1])
    return perceptron(w, b, x)

def perceptron_network(input_val):
    o1 = and_function(input_val)
    o2 = or_function(input_val)
    o3 = not_function(o1)
    o4 = np.array([o2, o3])
    o5 = and_function(o4)
    return o5

print("(0, 0): " + str(perceptron_network(np.array([0, 0]))))
print("(0, 1): " + str(perceptron_network(np.array([0, 1]))))
print("(1, 0): " + str(perceptron_network(np.array([1, 0]))))
print("(1, 1): " + str(perceptron_network(np.array([1, 1])))) 