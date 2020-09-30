# -*- coding: utf-8 -*-
# @Author: lenovouser
# @Date:   2020-09-30 16:54:52
# @Last Modified by:   lenovouser
# @Last Modified time: 2020-09-30 16:54:54
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:42:21 2020

@author: lenovouser
"""
import numpy as np

from matplotlib import pyplot as plt


class nn:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

        self.innodes = inputnodes
        self.hidnodes = hiddennodes
        self.outnodes = outputnodes

        self.train_e = [0]
        self.valid_e = [0]

        self.wih = np.random.normal(0, pow(self.hidnodes, -0.5),
                                    (self.hidnodes, self.innodes))

        self.who = np.random.normal(0,
                                    pow(self.outnodes, -0.5),
                                    (self.outnodes, self.hidnodes))

        self.acfun = lambda x: 1 / (1 + np.exp(-1 * x))

    def sgd_train(self, indexs, train_data_list):

        dwho = np.zeros([batch_size, self.who.shape[0], self.who.shape[1]])
        dwih = np.zeros([batch_size, self.wih.shape[0], self.wih.shape[1]])
        c = 0
        for l in train_data_list[indexs]:

            all_values = l.split(",")
            inputs = (np.asfarray(all_values[1:]) / 255 * 0.98) + 0.01
            targets = np.zeros(outputnodes) + 0.01
            targets[int(all_values[0])] = 0.99

            inputs = np.array(inputs, ndmin=2).T
            target_list = np.array(targets, ndmin=2).T

            hiddeninput = np.dot(self.wih, inputs)
            hidden_output = self.acfun(hiddeninput)

            outinput = np.dot(self.who, hidden_output)
            output = self.acfun(outinput)

            errors = target_list - output
            herrors = np.dot(self.who.T, errors * output * (1 - output))
            self.train_e[-1] += np.sum(errors**2)

            dwho[c] = np.dot(errors * output * (1 - output),
                             hidden_output.T)

            dwih[c] = np.dot(herrors * hidden_output * (1 - hidden_output),
                             inputs.T)
            c += 1

        self.who += self.lr * dwho.mean(0)
        self.wih += self.lr * dwih.mean(0)

    def query(self, inputs):
        inputs = np.array(inputs, ndmin=2).T
        hiddeninput = np.dot(self.wih, inputs)
        hidden_output = self.acfun(hiddeninput)

        outinput = np.dot(self.who, hidden_output)
        output = self.acfun(outinput)

        return output


inputnodes = 28 * 28
hiddennodes = 300
outputnodes = 10

learningrate = 0.1
batch_size = 5

n = nn(inputnodes, hiddennodes, outputnodes, learningrate)
train_data_file = open("mnist_train_100.csv", "r")
num = 100
train_num = int(num * 0.6)

train_data_list = train_data_file.readlines()
train_data_file.close()

epoch = 10
for i in range(epoch):
    rn = np.random.permutation(num)
    train_data_list = np.array(train_data_list)
    rn = rn.reshape(-1, batch_size)

    n.lr = learningrate * np.exp(-i * 0.1)
    for l in range(train_num // batch_size):
        n.sgd_train(rn[l], train_data_list)

    n.train_e.append(0)

    for l in train_data_list[rn[train_num // batch_size]]:
        all_values = l.split(",")
        inputs = (np.asfarray(all_values[1:]) / 255 * 0.98) + 0.01
        targets = np.zeros(outputnodes) + 0.01
        targets[int(all_values[0])] = 0.99

        n.valid_e[-1] += np.sum((targets - n.query(inputs))**2)
    n.valid_e.append(0)

n.train_e = np.array(n.train_e[:epoch]) / train_num * batch_size
n.valid_e = np.array(n.valid_e[:epoch]) / (num - train_num) * batch_size
x = list(range(epoch))

plt.plot(x, n.train_e, color='r', label="train")
plt.plot(x, n.valid_e, color="b", label="valid")
plt.legend()
plt.show()
test_data_file = open("mnist_test_10.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()
score = []
for l in test_data_list:
    all_values = l.split(",")
    # print(all_values)
    label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255 * 0.98) + 0.01
    outputs = n.query(inputs)
    if label == outputs.argmax():
        score.append(1)
    else:
        score.append(0)
print(sum(score) / len(score))
