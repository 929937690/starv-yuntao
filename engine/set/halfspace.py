#!/usr/bin/python3
import numpy as np

class HalfSpace:
    # HalfSpace class defining Gx <= g

    # constructor
    def __init__(obj,
                 G = np.matrix([]), # half-space matrix
                 g = np.matrix([]),  # half-space vector
                 dim = 0 # dimension of half-space
                 ):

        assert isinstance(G, np.ndarray), 'error: G matrix is not an ndarray'
        assert isinstance(g, np.ndarray), 'error: g matrix is not an ndarray'

        [n1, m1] = G.shape
        [n2, m2] = g.shape

        if n1 != n2:
            'error: Inconsistent dimension between half-space matrix and half-space vector'

        if m2 != 1:
            'error: Half-space vector should have one column'

        obj.G = G
        obj.g = g
        obj.dim = m1

        return

    def __repr__(obj):
        return "class: %s \nG: %s \ng: %s \ndim: %s " % (obj.__class__, obj.G, obj.g, obj.dim)
