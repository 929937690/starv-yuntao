import numpy as np

from engine.nn.funcs.poslin import PosLin
from engine.nn.layers.layer import Layers
from engine.set.star import Star
from engine.nn.fnn.FFNNS import FFNNS
from engine.set.halfspace import HalfSpace

# ----------------- test for FFNNS isSafe----------------------

W = np.matrix('1 1;'
              '0 1')
b = np.matrix('0;'
              '0.5')
L = Layers(W, b, 'poslin')
#print(L.__repr__())

Layers = []
Layers.append(L)
#print(Layers)
F = FFNNS(Layers) # network
#print(F.__repr__())

lb = np.matrix('-1; -1')
ub = np.matrix('1; 1')
I = Star(lb=lb, ub=ub) # input set

[R, time] = F.reach(I, 'exact-star')
G = np.matrix('-1 0')
g = np.matrix('-1.5')

U = HalfSpace(G, g) # unsafe region
#print(U.__repr__())
n_samples = 100
U_list = []
U_list.append(U)
#[safe, t, counter_inputs] = F.isSafe(I, U_list, 'exact-star')
#[safe, t, counter_inputs] = F.isSafe(I, U_list, 'approx-star', n_samples)
#[safe, t, counter_inputs] = F.isSafe(I, U_list, 'approx-zono', n_samples)

safe = 0
t = 1.4782
counter_inputs = np.matrix('0.692960399493948 0.944822497116195 0.941135111490313;'
                           '0.889683301165989 0.938803958454275 0.817350086011257')
counter_outputs = F.sample(counter_inputs)
print(counter_outputs)