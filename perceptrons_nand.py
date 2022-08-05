import sys
import ast

def perceptron(A, w, b, x):
    sum = 0
    for f, o in zip(w, x):
        sum+=int(f)*int(o)
    return A(sum+float(b))

def step(num):
    if num>0:
        return 1
    else:
        return 0

def perceptron_network(input_val):
    o3 = perceptron(step, (-1, -1), 2, input_val)
    return o3

x = ast.literal_eval(sys.argv[1])
output = perceptron_network(x)
print(output)