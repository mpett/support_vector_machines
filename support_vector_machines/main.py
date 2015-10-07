__author__ = 'martinpettersson'
from cvxopt.solvers import qp
from cvxopt.base import matrix
import pylab, random, math
import numpy as np

def my_kernel(x,y):
    return np.dot(x, y.T)

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

generate_datapoints()