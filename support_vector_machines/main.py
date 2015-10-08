__author__ = 'martinpettersson'
from cvxopt.solvers import qp
from cvxopt.base import matrix
import pylab, random, math
import numpy as np

def linear_kernel(x,y):
    return np.dot(x, y.T) + 1

def polynomial_kernel(x,y,p):
    return math.pow((np.dot(x, y.T) + 1), p)

def radial_basis_function_kernel(x,y,sigma):
    exponent = math.pow(x-y,2) / math.pow(2*sigma,2)
    return math.exp(-exponent)

def sigmoid_kernel(x,y,k,delta):
    return math.tanh(k*np.dot(x.T,y) - delta)

def generate_datapoints():
    classA = [(random.normalvariate(-1.5, 1),
               random.normalvariate(0.5, 1), 1.0)
              for i in range(5)] + \
                [(random.normalvariate(1.5,1),
                  random.normalvariate(0.5,1), 1.0)
                 for i in range(5)]
    classB = [(random.normalvariate(0.0,0.5),
                  random.normalvariate(-0.5,0.5), -1.0)
                 for i in range(10)]
    data = classA + classB
    random.shuffle(data)
    plot_datapoints(classA,classB)

def plot_datapoints(classA, classB):
    pylab.hold(True)
    pylab.plot([p[0] for p in classA],
               [p[1] for p in classA],
               'bo')
    pylab.plot([p[0] for p in classB],
               [p[1] for p in classB],
               'ro')
    pylab.show()

generate_datapoints()