#!/usr/bin/python3
import numpy as np

from engine.set.star import Star
from engine.nn.funcs.poslin import PosLin

class Operation:
    # Operation : class for specific operation in Feedforward neural network reachability

    # An Operation can be:
    #   1) AffineMap
    #   2) PosLin_stepExactReach
    #   3) PosLin_approxReachZono
    #   4) PosLin_approxReachAbsDom
    #   5) PosLin_approxReachStar

    #   6) SatLin_approxReachStar
    #   7) SatLin_stepExactReach
    #   8) SatLin_approxReachZono
    #   9) SatLin_approxReachAbsDom

    #   10) SatLins_stepExactReach
    #   11) SatLins_approxReachStar
    #   12) SatLins_approxReachZono
    #   13) SatLins_approxReachAbsDom

    #   14) LogSig_approxReachZono
    #   15) LogSig_approxReachStar

    #   16) TanSig_approxReachZono
    #   17) TanSig_approxReachStar

    # The main method for Operation class is:
    #   2) execute
    # The Operation class is used for verification of FFNN using Deep First
    # Search algorithm

    def __init__(obj,
                 Name = '',
                 map_mat = np.matrix([]), #  used if the operation is an affine mapping operation
                 map_vec = np.matrix([]), # used if the operation is an affine mapping operation

                 index = np.matrix([]), #  used if the operation is PosLin or SatLin or SatLins stepReach operation
                 # index is the index of the neuron the stepReach is performed

                 method = '' # reach method
    ):

        # ---------- Constructor --------
        # @name: name of the operation
        # @W: affine mapping matrix
        # @b: affine mapping vector
        # @index: neuron index

        assert isinstance(map_mat, np.ndarray), 'error: map_mat matrix is not an ndarray'
        assert isinstance(map_vec, np.ndarray), 'error: map_vec matrix is not an ndarray'
        assert isinstance(index, np.ndarray), 'error: index matrix is not an ndarray'

        if Name and map_mat.size and map_vec.size:
            if Name != 'AffineMap':
                'error: The operation is not an affine mapping operation'

            if map_mat.shape[0] != map_vec.shape[0]:
                'error: Inconsistent dimension between that affine mapping matrix and vector'

            if map_vec.shape[1] != 1:
                'error: Affine mapping vector should have one column'

            obj.Name = Name
            obj.map_mat = map_mat
            obj.map_vec = map_vec

        elif Name and index.size:
            if Name != 'PosLin_stepExactReach' and Name != 'SatLin_stepExactReach' and Name != 'SatLins_stepExactReach':
                'error: Unknown operation name'
            if index < 1:
                'error: Invalid neuron index'

            obj.Name = Name
            obj.index = index

        elif Name:
            S1 = (Name != 'PosLin_approxReachStar' and Name != 'PosLin_approxReachZono' and Name != 'PosLin_approxReachAbsDom')
            S2 = (Name != 'SatLin_approxReachStar' and Name != 'SatLin_approxReachZono' and Name != 'SatLin_approxReachAbsDom')
            S3 = (Name != 'SatLins_approxReachStar' and Name != 'SatLins_approxReachZono' and Name != 'SatLins_approxReachAbsDom')
            S4 = (Name != 'LogSig_approxReachStar' and Name != 'LogSig_approxReachZono')
            S5 = (Name != 'TanSig_approxReachStar' and Name != 'TanSig_approxReachZono')

            if S1 and S2 and S3 and S4 and S5:
                'error: Unkown operation name'
            obj.Name = Name
        else:
            'error: Invalid number of arguments'

        obj.method = method
        return

    # execute the operation
    def execute(obj, I):
        # @I: a star input set
        # @S: a star output set or an array of star output sets

        assert isinstance(I, Star), 'error: input set is not a star set'

        if obj.Name == 'AffineMap':
            S = I.affineMap(obj.map_mat, obj.map_vec)
        # PosLin
        elif obj.Name == 'PosLin_stepExactReach':
            [xmin, xmax] = I.estimateRange(obj.index)
            S = PosLin.stepReach(I, obj.index, xmin, xmax)
        elif obj.Name == 'PosLin_approxReachStar':
            S = PosLin.reach_star_approx(I)
        elif obj.Name == 'PosLin_approxReachZono':
            S = PosLin.reach_zono_approx(I)
        else:
            'error: Unknown operation'
        return S

