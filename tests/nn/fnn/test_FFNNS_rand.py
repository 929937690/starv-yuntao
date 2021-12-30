import numpy as np

from engine.nn.funcs.poslin import PosLin
from engine.nn.layers.layer import Layers
from engine.set.star import Star
from engine.nn.fnn.FFNNS import FFNNS
from engine.set.halfspace import HalfSpace

# --------------------- test for FFNNS rand function
neurons = np.matrix('2 3 3 4')
funcs = ['poslin', 'poslin', 'poslin']
net = FFNNS.rand(neurons, funcs)

V = np.matrix('0 0; 1 0; 0 1')
I_A = np.matrix('-0.0813237059702952 -0.608500284572824;'
                '0.502136038925144 -0.489923109941502;'
                '0.705859832761647 -0.0151477944468789;'
                '0.162582280130934 0.419003561972952;'
                '-0.523719632178512 0.256938524779700;'
                '-0.687228235004730 -0.116310542482995')
I_b = np.matrix('0.789375613077861;'
                '0.712625248470620;'
                '0.708189551474105;'
                '0.893310146165040;'
                '0.812219392377968;'
                '0.717069878547129')
lb = np.matrix('-1.17357831575357;'
               '-1.31539495464738')
ub = np.matrix('1.04038982580404;'
               '2.29657079706661')
I = Star(V=V.transpose(), C=I_A, d=I_b, lb=lb, ub=ub)
#X = I.sample(10)
X = np.matrix('0.656055303574990 0.603912824992392 -0.714420994010226 -0.590819760968028 -0.0923704414342075 0.525928848685459 -0.392334181661267 -0.327731938705958 0.976609857684336 -0.147178086089669;'
              '-0.646086201046897 -0.206390682190954 0.984340081007819 -0.739424682332099 -0.999339774506302 1.43641670978782 1.24232042654956 -0.878841322893156 0.0537611576611499 0.801433091877643')

[S, time] = net.reach(I, 'exact-star')
Y = net.sample(X)