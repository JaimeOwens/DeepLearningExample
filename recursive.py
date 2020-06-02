#!/usr/bin/env python2
# -*- coding: UTF-8 -*-

import numpy as np
from activators import IdentityActivator

class TreeNode(object):
    def __init__(self, data, children=[], children_data=[]):
        self.parent = None
        self.children = children
        self.children_data = children_data
        self.data = data
        for child in children:
            child.parent = self

class RecursiveLayer(object):
    def __init__(self, node_width, child_count, 
                 activator, learning_rate):
        self.node_width = node_width
        self.child_count = child_count
        self.activator = activator
        self.learning_rate = learning_rate
        self.W = np.random.uniform(-1e-4, 1e-4,
            (node_width, node_width * child_count))
        self.b = np.zeros((node_width, 1))
        self.root = None

    def forward(self, *children):
        children_data = self.concatenate(children)
        parent_data = self.activator.forward(
            np.dot(self.W, children_data) + self.b
        )
        self.root = TreeNode(parent_data, children
                            , children_data)

    def backward(self, parent_delta):
        self.calc_delta(parent_delta, self.root)
        self.W_grad, self.b_grad = self.calc_gradient(self.root)

    def update(self):
        self.W -= self.learning_rate * self.W_grad
        self.b -= self.learning_rate * self.b_grad

    def reset_state(self):
        self.root = None

    def concatenate(self, tree_nodes):
        concat = np.zeros((0,1))
        for node in tree_nodes:
            concat = np.concatenate((concat, node.data))
        return concat

    def calc_delta(self, parent_delta, parent):
        parent.delta = parent_delta
        if parent.children:
            children_delta = np.dot(self.W.T, parent_delta) * (
                self.activator.backward(parent.children_data)
            )
            slices = [(i, i * self.node_width, 
                        (i + 1) * self.node_width)
                        for i in range(self.child_count)]
            for s in slices:
                self.calc_delta(children_delta[s[1]:s[2]], 
                                parent.children[s[0]])

    def calc_gradient(self, parent):
        W_grad = np.zeros((self.node_width, 
                            self.node_width * self.child_count))
        b_grad = np.zeros((self.node_width, 1))
        if not parent.children:
            return W_grad, b_grad
        parent.W_grad = np.dot(parent.delta, parent.children_data.T)
        parent.b_grad = parent.delta
        W_grad += parent.W_grad
        b_grad += parent.b_grad
        for child in parent.children:
            W, b = self.calc_gradient(child)
            W_grad += W
            b_grad += b
        return W_grad, b_grad

    def dump(self, **kwArgs):
        print 'root.data: %s' % self.root.data
        print 'root.children_data: %s' % self.root.children_data
        if kwArgs.has_key('dump_grad'):
            print 'W_grad: %s' % self.W_grad
            print 'b_grad: %s' % self.b_grad


def data_set():
    children = [
        TreeNode(np.array([[1],[2]])),
        TreeNode(np.array([[3],[4]])),
        TreeNode(np.array([[5],[6]]))
    ]
    d = np.array([[0.5],[0.8]])
    return children, d


def gradient_check():
    error_function = lambda o: o.sum()
    rnn = RecursiveLayer(2, 2, IdentityActivator(), 1e-3)
    x, d = data_set()
    rnn.forward(x[0], x[1])
    rnn.forward(rnn.root, x[2])
    sensitivity_array = np.ones((rnn.node_width, 1),
                                dtype=np.float64)
    rnn.backward(sensitivity_array)
    epsilon = 10e-4
    for i in range(rnn.W.shape[0]):
        for j in range(rnn.W.shape[1]):
            rnn.W[i,j] += epsilon
            rnn.reset_state()
            rnn.forward(x[0], x[1])
            rnn.forward(rnn.root, x[2])
            err1 = error_function(rnn.root.data)
            rnn.W[i,j] -= 2*epsilon
            rnn.reset_state()
            rnn.forward(x[0], x[1])
            rnn.forward(rnn.root, x[2])
            err2 = error_function(rnn.root.data)
            expect_grad = (err1 - err2) / (2 * epsilon)
            rnn.W[i,j] += epsilon
            print 'weights(%d,%d): expected - actural %.4e - %.4e' % (
                i, j, expect_grad, rnn.W_grad[i,j])
    return rnn


def test():
    children, d = data_set()
    rnn = RecursiveLayer(2, 2, IdentityActivator(), 1e-3)
    rnn.forward(children[0], children[1])
    rnn.dump()
    rnn.forward(rnn.root, children[2])
    rnn.dump()
    rnn.backward(d)
    rnn.dump(dump_grad='true')
    return rnn

if __name__ == '__main__':
    test()
    gradient_check()
