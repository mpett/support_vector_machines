__author__ = 'martinpettersson'
from cvxopt.solvers import qp
from cvxopt.base import matrix
import pylab, random, math
import numpy as np

def linear_kernel(x,y):
    return np.dot(x, np.transpose(y)) + 1

def polynomial_kernel(x,y,p):
    return math.pow((np.dot(x, np.transpose(y)) + 1), p)

def radial_basis_function_kernel(x,y,sigma):
    exponent = math.pow(x-y,2) / math.pow(2*sigma,2)
    return math.exp(-exponent)

def sigmoid_kernel(x,y,k,delta):
    return math.tanh(k*np.dot(np.transpose(x),y) - delta)

def indicator(x,y):
    return []

def generate_p_matrix(data):
    p_matrix = [[0 for element in range(len(data))] for element in range(len(data))]
    t_values = [0 for element in range(len(data))]
    vectors = [0 for element in range(len(data))]
    index = 0
    for element in data:
        t_values[index] = element[2]
        vectors[index] = element[:2]
        index+=1
    for i in range(len(data)):
        for j in range(len(data)):
            p_matrix[i][j] = t_values[i]*t_values[j]*linear_kernel(vectors[i],vectors[j])
    return p_matrix

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
    #plot_datapoints(classA,classB)
    return data

def plot_datapoints(classA, classB):
    pylab.hold(True)
    pylab.plot([p[0] for p in classA],
               [p[1] for p in classA],
               'bo')
    pylab.plot([p[0] for p in classB],
               [p[1] for p in classB],
               'ro')
    pylab.show()

def plot_decision_boundary():
    xrange = np.arange(-4,4,0.05)
    yrange = np.arange(-4,4,0.05)
    grid = matrix([[indicator(x,y) for y in yrange] for x in xrange])
    pylab.contour(xrange,yrange,grid,
                  (-1.0,0.0,1.0),
                  colors=('red','black','blue'),
                  linewidths=(1,3,1))

def main():
    data = generate_datapoints()
    N = len(data)

main()