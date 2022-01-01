import numpy as np
import scipy.io

from engine.set.star import Star
from engine.tools.precision import Precision
from engine.nn.fnn.FFNNS import FFNNS

mat_content = scipy.io.loadmat('NeuralNetwork7_3.mat')
W = mat_content['W']
b = mat_content['b']
#print(mat_content)

Layers = []
n = len(b[0])
p = Precision()
for i in range(n-1):
    print(b[i])
    bi = p.cell2mat(b[i])
    Wi = p.cell2mat(W[i])
    Li = Layers(Wi, bi, 'poslin')
    Layers.append(Li)

bn = p.cell2mat(b[n])
Wn = p.cell2mat(W[n])
Ln = Layers(Wn, bn, 'poslin')

Layers.append(Ln)

F = FFNNS(Layers)

C = np.matrix('1 0 0 ;'
              '-1 0 0;'
              '0 1 0;'
              '0 -1 0;'
              '0 0 1;'
              '0 0 -1')
d = np.matrix('1; 1; 1; 1; 1; 1')
V = np.matrix('0 0 0;'
              '1 0 0;'
              '0 1 0;'
              '0 0 1')
I = Star(V.transpose(), C, d) # input set as a Star set

# select option for reachability algorithm

[R, t] = F.reach(I); # compute reach set using stars and 4 cores
F.printtoConsole() # print all information to console
scipy.io.savemat('F.mat', F)