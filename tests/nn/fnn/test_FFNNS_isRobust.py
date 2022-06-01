import numpy as np

from engine.nn.funcs.poslin import PosLin
from engine.nn.Layer.layer import Layer
from engine.set.star import Star
from engine.nn.fnn.FFNNS import FFNNS
from engine.set.halfspace import HalfSpace

# ----------------- test for FFNNS isRobust ----------------------

W = np.matrix('1 1;'
              '0 1')
b = np.matrix('0;'
              '0.5')
L = Layer(W, b, 'poslin')
#print(L.__repr__())

Layer = []
Layer.append(L)
#print(Layer)
F = FFNNS(Layer) # network
#print(F.__repr__())

input_vec = np.matrix('1; 1') # input vector for a single points
dis_bound = 0.5 # disturbance bound

G = np.matrix('-1 0')
g = np.matrix('-1.5')

U = HalfSpace(G, g) # unsafe region
#print(U.__repr__())
U_list = []
U_list.append(U)

n_samples = 100

[robust1, t1, counter_inputs1] = F.isRobust(input_vec, dis_bound, U_list, 'exact-star')

[robust2, t2, counter_inputs2] = F.isRobust(input_vec, dis_bound, U_list, 'approx-zono', n_samples)

counter_outputs = F.sample(counter_inputs2)
print(counter_outputs)