#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from __future__ import print_function
from functools import reduce

class VectorOp(object):
    @staticmethod
    def dot(x, y):
        return reduce(lambda a, b: a + b, VectorOp.element_multiply(x, y), 0.0)

    @staticmethod
    def element_multiply(x, y):
        return list(map(lambda x_y: x_y[0] * x_y[1], zip(x, y)))      

    @staticmethod
    def element_add(x, y):
        return list(map(lambda x_y: x_y[0] + x_y[1], zip(x, y)))

    @staticmethod
    def scala_multiply(v, s):
        return map(lambda e: e*s, v)

class Perceptron(object):
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0.0] * input_num
        self.bias = 0.0

    def __str__(self):
        return 'weights\t:%s\nbias:\t:%f\n' %(self.weights, self.bias)

    def predict(self, input_vec):
        return self.activator(
            VectorOp.dot(input_vec, self.weights) + self.bias
        )

    def train(self, input_vecs, labels, iteration, rate):
        for _ in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        samples = zip(input_vecs, labels)
        for(input_vec, label) in samples:
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        delta = label - output
        self.weights = VectorOp.element_add(
            self.weights, VectorOp.scala_multiply(input_vec, rate * delta)
        )
        self.bias += rate * delta

def f(x):
    return 1 if x > 0 else 0


def get_training_dataset():
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 0, 0]
    return input_vecs, labels


def train_and_perceptron():
    p = Perceptron(2, f)
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p

if __name__ == '__main__':
    and_perception = train_and_perceptron()
    print(and_perception)
    print('1 and 1 = %d' % and_perception.predict([1, 1]))
    print('0 and 0 = %d' % and_perception.predict([0, 0]))
    print('1 and 0 = %d' % and_perception.predict([1, 0]))
    print('0 and 1 = %d' % and_perception.predict([0, 1]))