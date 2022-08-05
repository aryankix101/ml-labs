import numpy as np
import math
import pickle

def sigmoid(num):
    return (1/(1+math.e**(-num)))

#Forward pass
def network_generate(A, x, wList, bList):
    vA = np.vectorize(A)
    a = [x]
    for i in range(len(wList)):
        temp = vA(a[i] @ wList[i] + bList[i])
        a.append(temp)
    return a

#Back prop
def backProp(x, y, wList, bList, l):
    network = network_generate(sigmoid, x, wList, bList)
    final = network[-1]
    dN = final * (1 - final) * (y - final)
    deltaList = [dN]
    for i in range(len(wList)-1, -1, -1):
        dL = network[i] * (1 - network[i]) * (deltaList[0] @ (wList[i].transpose()))
        deltaList.insert(0, dL)
    for i in range(len(deltaList)-1):
        bList[i] = bList[i] + l*deltaList[i+1]
        wList[i] = wList[i] + l * (network[i].transpose()) @ deltaList[i+1]
    return (wList, bList)


wList = [2 * np.random.rand(784, 300) - 1, 2 * np.random.rand(300, 100) - 1, 2 * np.random.rand(100, 10) - 1]
bList = [2 * np.random.rand(1, 300) - 1, 2 * np.random.rand(1, 100) - 1, 2 * np.random.rand(1, 10) - 1]
 
inputs = []
outputs = []

with open("mnist_train.csv") as f:
    for line in f:
        line = line.split(",")
        for x in range(len(line)):
            line[x] = int(line[x])
        number = []
        for x in range(line[0]):
            number.append(0)
        number.append(1)
        while(len(number) < 10):
            number.append(0)
        output = np.array([number])
        inputVec = np.array([line[1:]])
        inputVec = inputVec/255
        outputs.append(output)
        inputs.append(inputVec)

testinputs = []
testoutputs = []
with open("mnist_test.csv") as f:
    for line in f:
        line = line.split(",")
        for x in range(len(line)):
            line[x] = int(line[x])
        number = []
        for x in range(line[0]):
            number.append(0)
        number.append(1)
        while(len(number) < 10):
            number.append(0)
        output = np.array([number])
        inputVec = np.array([line[1:]])
        inputVec = inputVec/255
        testoutputs.append(output)
        testinputs.append(inputVec)

for e in range(10):
    for i in range(len(inputs)):
        temp = backProp(inputs[i], outputs[i], wList, bList, 0.5)
        wList = temp[0]
        bList = temp[1]
    outfile = open("networkmatrices.txt", 'wb')
    pickle.dump((wList, bList), outfile)
    outfile.close()
    print(e)


print("Network Architecture: 784, 300, 100, 10")
misclassified = 0
for x in range(len(inputs)):
    outputVec = network_generate(sigmoid, inputs[x], wList, bList)[-1][0]
    result = np.where(outputVec == max(outputVec))[0][0]
    if(outputs[x][0][result] != 1):
        misclassified += 1
print("Percent misclassified in training set: " + str((misclassified/len(inputs)*100)) + "%")

misclassified = 0
for x in range(len(testinputs)):
    outputVec = network_generate(sigmoid, testinputs[x], wList, bList)[-1][0]
    result = np.where(outputVec == max(outputVec))[0][0]
    if(testoutputs[x][0][result] != 1):
        misclassified += 1
print("Percent misclassified in test set: " + str((misclassified/len(testinputs)*100)) + "%")