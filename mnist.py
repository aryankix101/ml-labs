# -*- coding: utf-8 -*-

import math

import numpy as np

import pickle

def p_net(A, x, w, b):
    lst = [x]
    nA = np.vectorize(A)
    length = len(w)
    for ind in range(length):
        temp = nA(lst[ind] @ w[ind] + b[ind])
        lst.append(temp)
    return lst

def sigmoid(number):
    return 1/(math.exp(-number)+1)

def backProp(x, y, wl, bl, l):
    ntwrk = p_net(sigmoid, x, wl, bl)
    fin = ntwrk[-1]
    dN = (y - fin) * fin * (1 - fin)
    dList = [dN]
    for w in range(len(wl)-1, -1, -1):
        dList.insert(0, ntwrk[w] * (1 - ntwrk[w]) * (dList[0] @ (wl[w].transpose())))
    for d in range(len(dList)-1):
        wl[d] = wl[d] + l * (ntwrk[d].transpose()) @ dList[d+1]
        bl[d] = (dList[d+1] * l) + bl[d]
    return (wl, bl)

b1 = np.random.rand(1, 300) * 2 - 1
b2 = np.random.rand(1, 100) * 2 - 1
b3 = np.random.rand(1, 10) * 2 - 1
bList = [b1, b2, b3]
w1 = np.random.rand(784, 300) * 2 - 1
w2 = np.random.rand(300, 100)* 2 - 1
w3 = np.random.rand(100, 10) * 2 - 1
wList = [w1, w2, w3]
num_epoch = 10
inpu = list()
outpu = list()
with open("mnist_train.csv") as f:
    for lines in f:
        line = lines.split(",")
        for l in range(len(line)):
            line[l] = int(line[l])  
        num = list()
        done = False
        while(num_epoch > len(num)):
            if done == False:
                for i in range(line[0]):
                    num.append(0)
                    if i + 1 == line[0]:
                        num.append(1)
                done = True
            else:
                num.append(0)
        while(num_epoch > len(num)):
            num.append(0)
        ivector = (np.array([line[1:]]))/255
        output = np.array([num])
        inpu.append(ivector)
        outpu.append(output)
    print("Training set read.")

testinpu = list()
testoutpu = list()
with open("mnist_test.csv") as f:
    for lines in f:
        line = lines.split(",")
        for l in range(len(line)):
            line[l] = int(line[l])
        num = list()
        done = False
        while(num_epoch > len(num)):
            if done == False:
                for i in range(line[0]):
                    num.append(0)
                    if i + 1 == line[0]:
                        num.append(1)
                done = True
            else:
                num.append(0)
        ivector = (np.array([line[1:]]))/255
        output = np.array([num])
        testinpu.append(ivector)
        testoutpu.append(output)
    print("Testing set read.")
    
for epoch in range(num_epoch):
    for i in range(len(inpu)):
        temp = backProp(inpu[i], outpu[i], wList, bList, 0.5)
        wList = temp[0]
        bList = temp[1]
    outfile = open("ntwrkmatrix.txt", 'wb')
    pickle.dump((wList, bList), outfile)
    outfile.close()
    print(epoch)
    
print("Network Architecture: 784, 300, 100, 10")
training_misclassified = 0
for x in range(len(inpu)):
    outVec = p_net(sigmoid, inpu[x], wList, bList)[-1][0]
    result = np.where(max(outVec) == outVec)[0][0]
    if not (outpu[x][0][result] == 1):
        training_misclassified += 1
print("Percent misclassified in training set: " + str((training_misclassified/len(inpu)*100)) + "%")

test_misclassified = 0
for x in range(len(testinpu)):
    outVec_test = p_net(sigmoid, testinpu[x], wList, bList)[-1][0]
    result = np.where(outVec_test == max(outVec_test))[0][0]
    if not (testoutpu[x][0][result] == 1):
        test_misclassified += 1
print("Percent misclassified in test set: " + str((test_misclassified/len(testinpu)*100)) + "%")

print("Epochs trained: " + str(num_epoch))