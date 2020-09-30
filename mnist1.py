# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:42:21 2020

@author: lenovouser
"""
import numpy as np


class nn:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.innodes = inputnodes
        self.hidnodes = hiddennodes
        self.outnodes = outputnodes
        self.lr = learningrate

        self.wih = np.random.normal(0, pow(self.hidnodes, -0.5),
                                    (self.hidnodes, self.innodes))
        self.who = np.random.normal(0,
                                    pow(self.outnodes, -0.5),
                                    (self.outnodes, self.hidnodes))
        self.acfun = lambda x: 1 / (1 + np.exp(-1 * x))

    def train(self, inputs_lists, target_list):
        inputs = np.array(inputs_lists, ndmin=2).T
        target_list = np.array(target_list, ndmin=2).T

        hiddeninput = np.dot(self.wih, inputs)
        hidden_output = self.acfun(hiddeninput)

        outinput = np.dot(self.who, hidden_output)
        output = self.acfun(outinput)

        errors = target_list - output

        self.who += self.lr * np.dot(errors * output * (1 - output),
                                     hidden_output.T)
        # 重新更新输出和误差
        outinput = np.dot(self.who, hidden_output)
        output = self.acfun(outinput)

        errors = target_list - output

        herrors = np.dot(self.who.T, errors)

        self.wih += self.lr * np.dot(herrors * hidden_output * (1 - hidden_output),
                                     inputs.T)

    def query(self, inputs):
        inputs = np.array(inputs, ndmin=2).T
        hiddeninput = np.dot(self.wih, inputs)
        hidden_output = self.acfun(hiddeninput)

        outinput = np.dot(self.who, hidden_output)
        output = self.acfun(outinput)

        return output


inputnodes = 28 * 28
hiddennodes = 200
outputnodes = 10

learningrate = 0.075
n = nn(inputnodes, hiddennodes, outputnodes, learningrate)
train_data_file = open("mnist_train_100.csv", "r")
train_data_list = train_data_file.readlines()
train_data_file.close()
epoch = 5
for i in range(epoch):
    for l in train_data_list:
        all_values = l.split(",")
        inputs = (np.asfarray(all_values[1:]) / 255 * 0.98) + 0.01
        targets = np.zeros(outputnodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
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
