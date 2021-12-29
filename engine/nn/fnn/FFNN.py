import numpy as np
from engine.nn.layers.layer import Layers
from engine.set.star import Star
from engine.set.box import Box
from engine.set.zono import Zono
from engine.set.imagestar import ImageStar

import time

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0  # initial time
    tf = time.time()  # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti  # returns the time difference

TicToc = TicTocGenerator()  # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Elapsed time: %f seconds.\n" % tempTimeInterval)

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

# -------------------Constructor, evaluation, sampling, print method
class FFNNS:
    # FFNNS Class is a new feedforward network class used to replace the old FFNN in the future
    # reachability analysis method: 'exact-star', 'approx-star'
    # author: Yuntao Li
    # date: 11/13/2021

    def __init__(obj,
                 Layers = np.matrix([]), # An array of Layers, eg, Layers = [L1 L2 ... Ln]
                 nL = 0, # number of Layers
                 nN = 0, # number of Neurons
                 nI = 0, # number of Inputs
                 nO = 0, # number of Outputs

                 # properties for each set computation

                 reachMethod = 'exact-star', # reachable set computation scheme, default - 'star'
                 reachOption = np.matrix([]), # parallel option, default - non-parallel computing
                 relaxFactor = 0, # use only for approximate star method, 0 mean no relaxation
                 numCores = 1, # number of cores (workers) using in computation
                 inputSet = np.matrix([]), # input set
                 reachSet = np.matrix([]), # reachable set of each layers
                 outputSet = np.matrix([]), # output reach set
                 reachTime = np.matrix([]), # computation time for each layers
                 totalReachTime = 0, # total computation time
                 numSample = 0, # default number of samples using to falsify a property
                 unsafeRegion = np.matrix([]), # unsafe region of the network
                 getCounterExs = 0, # default, not getting counterexamples

                 Operations = np.matrix([]), # flatten a network into a sequence of operations

                 dis_opt = np.matrix([]), # display option
                 lp_solver = 'gurobi', # lp solver option, should be 'gurobi'
                 Name = 'net' # default is 'net'
                 ):

        assert isinstance(reachOption, np.ndarray), 'error: reachOption matrix is not an ndarray'
        assert isinstance(Layers, np.ndarray), 'error: Layers matrix is not an ndarray'
        assert isinstance(inputSet, np.ndarray), 'error: inputSet matrix is not an ndarray'
        assert isinstance(reachSet, np.ndarray), 'error: reachSet vector is not an ndarray'
        assert isinstance(outputSet, np.ndarray), ' error: outputSet is not an ndarray'
        assert isinstance(reachTime, np.ndarray), 'error: reachTime is not an ndarray'
        assert isinstance(unsafeRegion, np.ndarray), 'error: unsafeRegion matrix is not an ndarray'
        assert isinstance(Operations, np.ndarray), 'error: Operations matrix is not an ndarray'
        assert isinstance(dis_opt, np.ndarray), 'error: dis_opt matrix is not an ndarray'

        if Layers.size:
            if Name:
                obj.Name = Name
            nL = np.size(Layers, 1) # number of Layer
            for i in range(nL):
                L = Layers[i]
                if not isinstance(L, Layers): 'error: Element of Layers array is not a Layer object'

            # check consistency between layers
            for i in range(nL-1):
                if not np.size(Layers[i].W, 1) == np.size(Layers[i+1].W, 2):
                    'error: Inconsistent dimensions between Layer and Layer'

            obj.Layers = Layers
            obj.nL = nL # number of layers
            obj.nI = np.size(Layers[i].W, 2) # number of inputs
            obj.nO = np.size(Layers[nL].W, 1) # number of outputs

            nN = 0
            for i in range(nL):
                nN = nN + Layers[i].N
            obj.nN = nN # number of neurons

            return
        raise Exception('error: failed to create FFNNS')

    # Evaluation of a FFNN
    def evaluate(obj, x):
        # Evaluation of this FFNN
        # @x: input vector x
        # @y: output vector y

        y = x
        for i in range(obj.nL):
            y = obj.Layers[i].evaluate(y)
        return

    # Sample of FFNN
    def sample(obj, V):
        # sample the output of each layer in the FFNN based on the vertices of input set I, this is useful for testing
        # @V: array of vertices to evaluate
        # @Y: output which is a cell array

        In = V
        for i in range(obj.nL):
            In = obj.Layers[i].sample(In)
        return

    # check if all activation functions are piece-wise linear
    def isPieceWiseNetwork(obj):
        n = obj.nL
        b = 1
        for i in range(obj.nL):
            f = obj.Layers[i].f
            if f != 'poslin' and f != 'purelin' and f != 'satlin' and f != 'satlins' and f != 'leakyrelu':
                b = 0
                return

    # print information to a file
    def printtoFile(obj, file_name):
        # @file_name: name of file you want to store all data information
        f = open(file_name, 'w')
        f.write('Feedforward Neural Network Information\n')
        f.write('\nNumber of layers: %d' % obj.nL)
        f.write('\nNumber of neurons: %d' % obj.nN)
        f.write('\nNumber of inputs: %d' % obj.nI)
        f.write('\nNumber of outputs: %d' % obj.nO)

        if obj.reachSet.size:
            f.write('\n\nReach Set Information')
            f.write('\nReachability method: %s' % obj.reachMethod)
            f.write('\nNumber of cores used in computation: %d' % obj.numCores)

            for i in range(len(obj.reachSet) - 1):
                f.write('\nLayer %d reach set consists of %d sets that are computed in %.5f seconds' % (i, obj.numReachSet[i], obj.reachTime[i]))

            f.write('\nOutput Layer reach set consists of %d sets that are computed in %.5f seconds' % (obj.numReachSet(obj.nL), obj.reachTime(obj.nL)))
            f.write('\nTotal reachable set computation time: %.5f' % obj.totalReachTime)

        f.close()

    # print information to console
    def printtoConsole(obj):

        print('Feedforward Neural Network Information\n')
        print('\nNumber of layers: %d' % obj.nL)
        print('\nNumber of neurons: %d' % obj.nN)
        print('\nNumber of inputs: %d' % obj.nI)
        print('\nNumber of outputs: %d' % obj.nO)

        if obj.reachSet.size:
            print('\n\nReach Set Information')
            print('\nReachability method: %s' % obj.reachMethod)
            print('\nNumber of cores used in computation: %d' % obj.numCores)

            for i in range(len(obj.reachSet) - 1):
                print('\nLayer %d reach set consists of %d sets that are computed in %.5f seconds' % (i, obj.numReachSet[i], obj.reachTime[i]))

            print('\nOutput Layer reach set consists of %d sets that are computed in %.5f seconds' % (obj.numReachSet(obj.nL), obj.reachTime(obj.nL)))
            print('\nTotal reachable set computation time: %.5f' % obj.totalReachTime)

    # start parallel pool for computing
    def start_pool(obj, numCores=1):
        # if (len(args) == 1):
        #     obj = args[0]
        #     nCores = obj.numCores
        # elif (len(args) == 2):
        #     obj = args[0]
        #     nCores = args[1]
        if numCores:
            nCores = numCores
        else: 'error: Invalid number of input arguments'

        if nCores > 1:
            print('Working on this........')
        return

#-------------------------------- reachability analysis method
    def reach(*args):
        # @I: input set, a star set
        # @method: = 'exact-star' or 'approx-star' -> compute reach set using stars
        #            'abs-dom' -> compute reach set using abstract
        #            'approx-zono' -> compute reach set using zonotope
        #            'domain' (support in the future)
        #            'face-latice' -> compute reach set using face-latice (support in the future)
        # @S: output set
        # @t: computation time
        # @numOfCores: number of cores you want to run the reachable set computation, @numOfCores >= 1, maximum is the number of
        # cores in your computer.

        # Author: Yuntao Li

        if (len(args) == 7):
            obj = args[0] # FFNNS object
            obj.inputSet = args[1] # input set
            obj.reachMethod = args[2] # reachability analysis method
            obj.numCores = args[3] # number of cores used in computation
            obj.relaxFactor = args[4]
            obj.dis_opt = args[5]
            obj.lp_solver = args[6]
        elif (len(args) == 6):
            obj = args[0] # FFNNS object
            obj.inputSet = args[1] # input set
            obj.reachMethod = args[2] # reachability analysis method
            obj.numCores = args[3] # number of cores used in computation
            obj.relaxFactor = args[4]
            obj.dis_opt = args[5]
        elif (len(args) == 5):
            obj = args[0] # FFNNS object
            obj.inputSet = args[1] # input set
            obj.reachMethod = args[2] # reachability analysis method
            obj.numCores = args[3] # number of cores used in computation
            obj.relaxFactor = args[4] # used only for approx-star method
        elif (len(args) == 4):
            obj = args[0] # FFNNS object
            obj.inputSet = args[1] # input set
            obj.reachMethod = args[2] # reachability analysis method
            obj.numCores = args[3] # number of cores used in computation
        elif (len(args) == 3):
            obj = args[0] # FFNNS object
            obj.inputSet = args[1] # input set
            obj.reachMethod = args[2] # reachability analysis method
            obj.numCores = 1
        elif (len(args) == 2):
            obj = args[0] # FFNNS object
            # check for isstruct and isfield method
            obj.inputSet = args[1] # input set
            obj.reachMethod = 'exact-star'
            obj.numCores = 1
        else: 'error: Invalid number of input arguments (should be 1, 2, 3, 4, 5, 6)'

        # if reachability analysis method is an over-approximate method, we use 1 core for computation
        if obj.reachMethod != 'exact-star':
            obj.numCores = 1

        # Zonotope method accepts both star and zonotop input set
        if obj.reachMethod == 'approx-zono':
            a = args[1]
            assert isinstance(a, Star), 'error: input set is not a star set'
            obj.inputSet = a.getZono()

        if obj.numCores == 1:
            obj.reachOption = np.array([]) # don't use parallel computing
        else:
            print('working on this......')

        obj.reachSet = np.array([1, obj.nL])
        obj.numReachSet = np.zeros([1, obj.nL]);
        obj.reachTime = np.array([]);

        # compute reachable set
        In = obj.inputSet

        for i in range (obj.nL):
            if obj.dis_opt == 'display':
                print('\nComputing reach set for Layer %d ...' % i)

            st = tic()
            In = obj.Layers[i].reach(In, obj.reachMethod, obj.reachOption, obj.relaxFactor, obj.dis_opt, obj.lp_solver)
            t1 = toc(st)

            obj.numReachSet[i] = len(In)
            obj.reachTime = np.column_stack([obj.reachTime, t1])

            if obj.dis_opt == 'display':
                print('\nExact computation time: %.5f seconds' % t1)
                print('\nNumber of reachable set at the output of layer %d: %d' % (i, len(In)))

        obj.outputSet = In
        obj.totalReachTime = np.sum(obj.reachTime)
        S = obj.outputSet
        t = obj.totalReachTime
        if obj.dis_opt == 'display':
            print('\nTotal reach set computation time: %.5f seconds' % obj.totalReachTime)
            print('\nTotal number of output reach sets: %d', len(obj.outputSet))

        return [S, t]

    def verify(*args):
        # 1: @I: input set, need to be a star set
        # 2: @U: unsafe region, a HalfSpace
        # 3: @method: = 'star' -> compute reach set using stars
        #            'abs-dom' -> compute reach set using abstract
        #            domain (support in the future)
        #            'face-latice' -> compute reach set using
        #            face-latice (support in the future)
        # 4: @numOfCores: number of cores you want to run the reachable
        # set computation, @numOfCores >= 1, maximum is the number of
        # cores in your computer.
        # 5: @n_samples : number of simulations used for falsification if
        # using over-approximate reachability analysis, i.e.,
        # 'approx-zono' or 'abs-dom' or 'abs-star'
        # note: n_samples = 0 -> do not do falsification
        #
        # @safe: = 1-> safe, 0-> unsafe, 2 -> unknown
        # @vt: verification time
        # @counterExamples: counterexamples
        #
        # Author: Yuntao Li

        if (len(args) == 2):
            obj = args[0]
            obj.inputSet = args[1]
            obj.reachMethod = 'exact-star'
            obj.numCores = 1
        elif (len(args) == 3):
            obj = args[0]
            obj.inputSet = args[1]
            obj.unsafeRegion = args[2]
        elif (len(args) == 4):
            obj = args[0]
            obj.inputSet = args[1]
            obj.unsafeRegion = args[2]
            obj.reachMethod = args[3]
        elif (len(args) == 5):
            obj = args[0]
            obj.inputSet = args[1]
            obj.unsafeRegion = args[2]
            obj.reachMethod = args[3]
            obj.numCores = args[4]
        elif (len(args) == 6):
            obj = args[0]
            obj.inputSet = args[1]
            obj.unsafeRegion = args[2]
            obj.reachMethod = args[3]
            obj.numCores = args[4]
            obj.numSamples = args[5]
        elif (len(args) == 7):
            obj = args[0]
            obj.inputSet = args[1]
            obj.unsafeRegion = args[2]
            obj.reachMethod = args[3]
            obj.numCores = args[4]
            obj.numSamples = args[5]
            obj.getCounterExs = args[6]
        else: 'error: Invalid number of inputs, should be 2 3 4 5 6 7'

        if obj.numSamples > 0:
            t = tic()
            print('\nPerform fasification with %d random simulations' % obj.numSamples)
            counterExample = obj.falsify(obj.inputSet, obj.unsafeRegion, obj.numSamples)
            vt = toc(t)
        else:
            counterExample = np.array([])

        if counterExample.size:
            safe = 0
            obj.outputSet = np.array([])
        else:
            if obj.numCores > 1:
                print('working on this.....')
            t = tic()
            # perform reachability analysis
            [R, time] = obj.reach(obj.inputSet, obj.reachMethod, obj.numCores)

            if obj.reachMetho == 'exact-star':
                n = len(R)
                counterExample = np.array([])
                safes = np.array([])
                G = obj.unsafeRegion.G
                g = obj.unsafeRegion.g
                getCEs = obj.getCounterExs
                V = obj.inputSet.V
                if obj.numCores > 1:
                    print('working on this....')
                else:
                    safe = 1
                    for i in range(n):
                        R1 = Star(R[1])
                        if R1.intersectHalfSpace(obj.unsafeRegion.G, obj.unsafeRegion.g).size:
                            if obj.getCounterExs and obj.reachMethod == 'exact-star':
                                counterExamples = np.column_stack([counterExamples, Star(V = obj.inputSet.V, C=R[i].C, d=R[i].d, pred_lb=R[i].predicate_lb, pred_ub=R[i].predicate_ub)])
                            else:
                                safe = 0
                                break
            else:
                safe = 1
                if obj.reachMethod != 'exact-polyhedron':
                    if obj.reachMethod == 'approx-zono':
                        R = R.toStar
                    if R.intersectHalfSpace(obj.unsafeRegion.G, obj.unsafeRegion.g).size:
                        safe = 1
                    else:
                        safe = 2
                else:
                    n = len(R)
                    for i in range (n):
                        R1 = Star(R[1])
                        if R1.intersectHalfSpace(obj.unsafeRegion.G, obj.unsafeRegion.g).size:
                            safe = 0
                            break
            vt = toc(t)

            return [safe, vt, counterExamples]

    # visualize verification results
    # plot the output sets in a specific direction
    def visualize(obj, proj_mat, proj_vec):
        # @proj_mat: projection matrix
        # @proj_vec: project vector
        print('working on this.....')

#----------------------input to output sensitivity
    def getMaxSensitiveInput(obj, I, output_mat):
        # @I: input, is a star set, or box, or zono
        # @output_mat: output matrix, y = C*x, x is the output vector of the networks
        # @maxSensInputId: the id of the input that is most sensitive with the output changes
        # @maxSensVal: percentage of sensitivity of all inputs

        assert isinstance(output_mat, np.ndarray), 'error: output_mat is not a ndarray'
        if not isinstance(I, Box):
            I1 = I.getBox
        else:
            I1 = I

        if I1.dim != obj.nI:
            'error: Inconsistency between the input set and the number of inputs in the networks'
        if obj.nO != output_mat.shape[1]:
            'error: Inconsistency between the output matrix and the number of outputs of the networks'

        maxSensVal = np.zeros(1, obj.nI)
        [R, time] = obj.reach(I1.toZono(), 'approx-zono')
        R = R.affineMap(output_mat, np.array([]))
        B = R.getBox
        max_rad = max(B.ub)

        for i in range (obj.nI):
            I2 = I1.singlePartition(1, 2) # partition into two boxes at index
            [R2, time] = obj.reach(I2[1].toZono, 'approx-zono')
            R2 = R2.affineMap(output_mat, np.array([]))
            B2 = R2.getBox
            max_rad2 = max(B2.ub)
            maxSensVal[i] = (max_rad - max_rad2) / max_rad

        maxSensInputId = max(maxSensVal)
        return maxSensInputId

    def partitionInput_MSG(obj, I, U):
        # @I: input, is a star set, or box, or zono
        # @U: unsafe region, a HalfSpace object
        # @Is: two partitioned inputs, the last one is close to the unsafe region
        #
        # Author: Yuntao Li

        if not isinstance(I, Box):
            I1 = I.getBox
        else:
            I1 = I

        if I1.dim != obj.nI:
            'error: Inconsistency between the input set and the number of inputs in the networks'
        # check if U is HalfSpace object

        if obj.nO != U.G.shape[1]:
            'error: Inconsistency between the output matrix and the number of outputs of the networks'

        maxSensVal = np.zeros(1, obj.nI)
        [R, time] = obj.reach(I1.toZono(), 'approx-zono')
        R = R.affineMap(U.G, -U.g)
        B = R.getBox
        max_rad = max(B.ub)

        for i in range (obj.nI):
            # mark
            I2 = I1.singlePartition(i, 2) # partition into two boxes at index i
            [R2, time] = obj.reach(I2[i].reach(I2[i].toZono, 'approx-zono'))
            R2 = R2.affineMap(U.G, -U.g)
            B2 = R2.getBox
            max_rad2 = max(B2.ub)
            maxSensVal[i] = (max_rad - max_rad2)/max_rad

        maxSensInputId = max(maxSensVal)

        # mark
        I1 = I.partition(maxSensInputId, 2)
        R1 = obj.reach(I1[1].toZono, 'approx-zono')
        R1 = R1.affineMap(U.G, -U.g)
        B1 = R1.getBox
        max_y1 = max(B1.ub)
        R2 = obj.reach(I1[2].toZono, 'approx-zono')
        R2 = R2.affineMap(U.G, -U.g)
        B2 = R2.getBox
        max_y2 = max(B2.ub)

        if max_y1 <= max_y2:
            Is = np.column_stack([I1[2], I1[1]])
        else:
            Is = np.column_stack([I1[1], I1[2]])

        return Is

    # depth first search for max sensitive inputs in partitioning
    def searchMaxSensInputs(obj, I, output_mat, k, maxSen_lb):
        # @I: input, is a star set, or box, or zono
        # @output_mat: output matrix, y = C*x, x is the output vector of the networks
        # @k: depth of search tree
        # @maxSen_lb: the search is stop if the max sensitive value is smaller than the maxSen_lb

        # @maxSensInputs: a sequence of input indexes that are most sensitive with the output changes
        # @maxSensVals: percentage of sensitivity corresponding to above sequence of inputs

        if k < 1: 'error: Invalid depth of search tree'
        if not isinstance(I, Box): 'error: Input is not a box'

        maxSensInputs = np.array([])
        maxSensVals = np.array([])
        for i in range (k):
            if i == 0:
                [max_id, sens_vals] = obj.getMaxSensitiveInput(I, output_mat)
                maxSensInputs = np.column_stack([maxSensVals, max(sens_vals)])
                I1 = I
            else:
                I2 = I1.singlePartition(maxSensInputs[i-1], 2)
                I2 = I2[1]
                [max_id, sens_vals] = obj.getMaxSensitiveInput(I2, output_mat)
                maxSensInputs = np.column_stack([maxSensInputs, max_id])
                maxSensVals = np.column_stack([maxSensVals, max(sens_vals)])
                I1 = I2

            if maxSen_lb.size and maxSensVals[i] < maxSen_lb:
                maxSensInputs[i] = np.array([])
                maxSensVals[i] = np.array([])
                break

        return [maxSensInputs, maxSensVals]

    # verify safety property using max sensitivity guided (MSG) method
    def verify_MSG(obj, I, reachMethod, k, sens_lb, U):
        # @I: input set, a box
        # @reachMethod: reachability method
        # @k: depth of search tree for max sensitive inputs
        # @sens_lb: lowerbound of sensitive value
        # @U: unsafe region, a halfspace object
        # @safe: = 1: safe
        #        = 2: unknown
        #        = 0: unsafe

        t = tic()
        [maxSensInputs, maxSensVals] = obj.searchMaxSensInputs(I, U.G, k, sens_lb)
        n = len(maxSensInputs)
        Is = I.partition(maxSensInputs, 2*np.ones(1, n))
        I1 = np.array([]) # set of partitioned inputs
        N = 2^n # number of partitioned inputs
        for i in range(N):
            if reachMethod == 'approx-zono':
                I1 = np.column_stack(I1, Is[i].toZono)
            elif reachMethod == 'approx-star' or reachMethod == 'abs-dom':
                I1 = np.column_stack(I1, Is[i].toStar)
            else:
                'error: reach method should be approx-zono or approx-star or abs-dom'

        safe_vec = np.zeros(1, N)
        [R, time] = obj.reach(I1, reachMethod) # perform reachability anlaysis
        for i in range(N):
            S = R[i].intersectHalfSpace(U.G, U.g)
            if S.size:
                safe_vec[i] = 1
            else:
                counterExamples = obj.falsify(I1[i], U, 1)
                if len(counterExamples) >= 1:
                    safe_vec[i] = 0
                    break
                else:
                    safe_vec[i] == 2

        if sum(safe_vec) == N:
            safe = 'SAFE'
            counterExamples = np.array([])
            print('\nThe Network is Safe!')
        else:
            if counterExamples.size:
                safe = 'UNSAFE'
                print('\nThe Network is Unsafe')
            else:
                safe = 'UNKNOWN'
                print('\nThe Safety of the Network is unknown')
        VT = toc(t)
        return [safe, VT, counterExamples]

    # get counter input candidate after a single partition using MSG
    def getStepCounterInputCand_MSG(obj, I, U):
        # @I: input, is a star set, or box, or zono
        # @output_mat: output matrix, y = C*x, x is the output vector of the networks
        # @k:
        # @maxSensInputId: the id of the input that is most sensitive with the output changes
        # @counterInputCand: counter input candidate

        if not isinstance(I, Box):
            I1 = I.getBox
        else:
            I1 = I

        if I1.dim != obj.nI:
            'error: Inconsistency between the input set and the number of inputs in the network'
        if obj.nO != U.G.shape[1]:
            'error: Inconsistency between the unsafe region and the number of outputs of the networks'

        maxSensVal = np.zeros(1, obj.nI)

        [R, time] = obj.reach(I1.toZono(), 'approx-zono')
        R = R.affineMap(U.G, -U.g)
        B = R.getBox
        max_rad = max(B.ub - B.lb)

        for i in range(obj.nI):
            I2 = I1.singlePartition(i, 2) # partition into two boxes at index i
            [R1, time] = obj.reach(I2[i].toZono, 'approx-zono')
            R1 = R1.affineMap(U.G, -U.g)
            B1 = R1.getBox
            max_rad1 = max(B1.ub - B1.lb)
            maxSensVal[i] = (max_rad - max_rad1)/max_rad

        maxSensInputId = max(maxSensVal)

        I2 = I1.singlePartition(maxSensInputId, 2)
        [R1, time] = obj.reach(I2[1].toZono, 'approx-zono')
        R1 = R1.affineMap(U.G, -U.g)
        B1 = R1.getBox
        max_y1 = max(B1.ub)

        [R2, time] = obj.reach(I2[2].toZono, 'approx-zono')
        R2 = R2.affineMap(U.G, -U.g)
        B2 = R2.getBox
        max_y2 = max(B2.ub)

        if max_y1 <= max_y2:
            counterInputCand = I2[1]
        else:
            counterInputCand = I2[2]

        return counterInputCand

    #  Depth First Search for Counter Input Candidate
    def searchCounterInputCand_MSG(obj, I, U, k):
        # @I: input, is a star set, or box, or zono
        # @U: unsafe region of the networks
        # @k: depth of search tree
        # @maxSensInputId: the id of the input that is most sensitive with the output changes
        # @counterInputCand: counter input candidate

        if k < 1:
            'error: depth of search tree should be >= 1'

        counterInputCand = I
        for i in range(k):
            counterInputCand = obj.getStepCounterInputCand_MSG(counterInputCand, U)
        return counterInputCand

    # Depth First Seach for Falsification using Maximum Sensitive
    # Guided Method
    def falsify_MSG(obj, I, U):
        # @I: input set, is a box
        # @U: unsafe region, a halfspace object
        # @counterExamples: counter example inputs
        counterInputs = np.array([])
        return counterInputs

    def verify_MSG2(obj, I, U):
        # @I: input set, is a box
        # @U: unsafe region, a halfspace object
        # @counterExamples: counter example inputs
        print('working on this....')

#------------------------- checking safety method or falsify safety property
    def isSafe(*args):
        # 1: @I: input set, need to be a star set
        # 2: @U: unsafe region, a set of HalfSpaces
        # 3: @method: = 'star' -> compute reach set using stars
        #            'abs-dom' -> compute reach set using abstract
        #            domain (support in the future)
        #            'face-latice' -> compute reach set using
        #            face-latice (support in the future)
        # 4: @numOfCores: number of cores you want to run the reachable
        # set computation, @numOfCores >= 1, maximum is the number of
        # cores in your computer.
        #
        # 5: @n_samples : number of simulations used for falsification if
        # using over-approximate reachability analysis, i.e.,
        # 'approx-zono' or 'abs-dom' or 'abs-star'
        # note: n_samples = 0 -> do not do falsification
        #
        # @safe: = 1 -> safe, = 0 -> unsafe, = 2 -> uncertain
        # @t : verification time
        # @counter_inputs

        # parse inputs
        if len(args) == 6:
            obj = args[0] # FFNNS objects
            I = args[1]
            U = args[2]
            method = args[3]
            n_samples = args[4]
            numOfCores = args[5]
        elif len(args) == 5:
            obj = args[0] # FFNNS objects
            I = args[1]
            U = args[2]
            method = args[3]
            n_samples = args[4]
            numOfCores = 1
        elif len(args) == 4:
            obj = args[0]  # FFNNS objects
            I = args[1]
            U = args[2]
            method = args[3]
            n_samples = 1000
            numOfCores = 1
        elif len(args) == 3:
            obj = args[0] # FFNNS objects
            I = args[1]
            U = args[2]
            method = 'exact-star'
            n_samples = 0
            numOfCores = 1
        else:
            'error: Invalid number of input arguments, should be 3, 4, 5, 6'

        start = tic()

        if U.size:
            'error: Please specify unsafe region using Half-space class'

        # performing reachability analysis
        [R, time] = obj.reach(I, method, numOfCores)

        # check safety
        n = len(R)
        m = len(U)
        R1 = np.array([])
        for i in range(n):
            if isinstance(R[i], Zono):
                B = R[i].getBox
                R1 = np.column_stack([R1, B.toStar]) # transform to star sets
            else:
                R1 = np.column_stack([R1, R[i]])

        violate_inputs = np.array([])
        if numOfCores == 1:
            for i in range(n):
                for j in range(m):
                    S = R1.intersectHalfSpace(U[j].G, U[j].g)
                    if S.size and method == 'exact-star':
                        I1 = Star(I.V, S.C, S.d) # violate input set
                        violate_inputs = np.column_stack([violate_inputs, I1])
                    else:
                        violate_inputs = np.column_stack([violate_inputs, S])
        elif numOfCores > 1:
            print('\nWorking on this......')

        if violate_inputs.size:
            safe = 1
            counter_inputs = np.array([])
            print('\nThe Network is Safe')
        else:
            if method == 'exact-star':
                safe = 0
                counter_inputs = violate_inputs # exact-method return complete counter input set
                print('\nThe Network is unsafe, counter inputs contains %d stars' % len(counter_inputs))
            else:
                if n_samples == 0:
                    print('\nDo not do falsification since n_samples = 0, you can choose to do falsification by set n_samples value > 0')
                    safe = 2
                    counter_inputs = np.array([])
                else:
                    counter_inputs = obj.falsify(I, U, n_samples)
                    if counter_inputs.size:
                        safe = 2
                        print('\nSafety is uncertain under using %d samples to falsify the network' % n_samples)
                        print('\nYou can try to increase the samples for finding counter inputs')
                    else:
                        safe = 0
                        print('\nThe network is unsafe, %d counter inputs are found using %d simulations' % (len(counter_inputs), n_samples))

        t = toc(start)
        return [safe, t, counter_inputs]

    # falsify safety property using random simulation
    def falsify(obj, I, U, n_samples):
        # @input: star set input
        # @U: unsafe region, a set of HalfSpaces
        # @n_samples: number of samples used in falsification
        # @counter_inputs: counter inputs that falsify the property

        counter_inputs = np.array([])
        if isinstance(I, Zono) or isinstance(I, Box):
            I1 = I.toStar()
        elif isinstance(I, Star):
            I1 = I
        else:
            'error: Unknown set representation'

        m = len(U)
        for i in range(m):
            if not isinstance(U[i], 'HalfSpace'):
                ('error %d^th unsafe region is not a HalfSpace' % i)

        if n_samples < 1:
            'error: Invalid number of samples'

        V = I1.sample(n_samples)
        n = V.shape[1] # number of samples

        for i in range(n):
            y = obj.evaluate(V[:, i])
            for j in range(m):
                if y in U[j]:
                    counter_inputs = np.column_stack([counter_inputs, V[:, i]])
        return counter_inputs

#------------------------------checking robustness and get robustness bound of feedforward networks
    # Problem description:
    # checking robustness of FFNN corresponding to an input and
    # L_infinity norm bound disturbance
    #
    # x is input, x' is a disturbed input such that ||x'- x|| <= e
    # y = FFNN(x), and y' = FFNN(x')
    # FFNN is called robust if y' is in a robust region RB(y') defined by user
    # for example, let F is image classification network
    # inputs of F is an 28x28 = 784 pixel images, F has one output with
    # range between 0 to 9 to reconize 10 digits, 0, 1, 2 ..., 9
    #
    # let consider digit one as an input x, we disturb digit one with
    # some bounded disturabnce e, we have ||x' - x|| <= e
    #
    # to be called robust, F need produce the output sastify 0.5 <= y' <= 1.5
    # If output is violated this specification, F may misclasify digit
    # one. Thefore the un-robust regions in this case is S1 = y' < 0.5 and S2 = y'> 1.5
    #
    # If robustness is not guaranteed, the method search for some
    # adverserial examples, i.e., find some x' that makes F violate the robustness property.

    def isRobust(*args):
        # 1: @input_vec: input vector x
        # 2: @dis_bound: disturbance bound
        # 3: @un_robust_reg: un-robust-region, a star
        # 4: @method: = 'exact-star' or 'approx-star' or 'approx-zono' or 'abs-dom'
        # 5: @lb_allowable: allowable lower bound of disturbed input:    lb_allowable(i) <= x'[i]
        # 6: @ub_allowable: allowable upper bound of disturbed output:    ub_allowable(i) >= x'[i]
        # x' is the disturbed vector by disturbance bound, |x' - x| <= dis_bound
        # x'[i] >= lb_allowable, x'[i] <= ub_allowable[i]
        # 7: @n_samples: number of samples used to find counter examples
        # if using over-approximate reachability analysis methods
        # 8: @numCores:  number of cores used in computation

        # @robust: = 1-> robust
        #        : = 0 -> unrobust
        #        : = 2 -> uncertain, cannot find counter example
        # @adv_inputs: adverserial inputs

        start = tic()
        # parse inputs
        if len(args) == 9:
            obj = args[0] # FFNNS object
            input_vec = args[1] # input vec
            dis_bound = args[2] # disturbance bound
            un_robust_reg = args[3] # ub-robust region
            method = args[4] # reachability analysis method
            lb_allowable = args[5] # allowable lower bound on disturbed inputs
            ub_allowable = args[6] # allowable upper bound on disturbed inputs
            n_samples = args[7] # number of samples used for finding counter examples
            num_cores = args[8] # number of cores used in computation

            # check consistency
            if lb_allowable.size and (len(lb_allowable) != len(ub_allowable)) or (len(lb_allowable) != len(input_vec)):
                'error: Inconsistent dimensions between allowable lower-, upper- bound vectors and input vector'
        elif len(args) == 8:
            obj = args[0]  # FFNNS object
            input_vec = args[1]  # input vec
            dis_bound = args[2]  # disturbance bound
            un_robust_reg = args[3]  # ub-robust region
            method = args[4]  # reachability analysis method
            lb_allowable = args[5]  # allowable lower bound on disturbed inputs
            ub_allowable = args[6]  # allowable upper bound on disturbed inputs
            n_samples = args[7]  # number of samples used for finding counter examples
            num_cores = 1  # number of cores used in computation
        elif len(args) == 6:
            obj = args[0]  # FFNNS object
            input_vec = args[1]  # input vec
            dis_bound = args[2]  # disturbance bound
            un_robust_reg = args[3]  # ub-robust region
            method = args[4]  # reachability analysis method
            lb_allowable = np.array([])  # allowable lower bound on disturbed inputs
            ub_allowable = np.array([])  # allowable upper bound on disturbed inputs
            n_samples = args[5]  # number of samples used for finding counter examples
            num_cores = 1  # number of cores used in computation
        elif len(args) == 5:
            obj = args[0]  # FFNNS object
            input_vec = args[1]  # input vec
            dis_bound = args[2]  # disturbance bound
            un_robust_reg = args[3]  # ub-robust region
            method = args[4]  # reachability analysis method
            lb_allowable = np.array([])  # allowable lower bound on disturbed inputs
            ub_allowable = np.array([])  # allowable upper bound on disturbed inputs
            n_samples = 1000  # number of samples used for finding counter examples
            num_cores = 1  # number of cores used in computation
        else:
            'error: Invalid number of input arguments (should be 4 or 5 or 8)'

        # construct input set
        n = len(input_vec)
        lb = input_vec
        ub = input_vec

        if len(args) == 8 or len(args) == 9:
            for i in range(n):
                if lb[i] - dis_bound > lb_allowable[i]:
                    lb[i] = lb[i] - dis_bound
                else:
                    lb[i] = lb_allowable[i]
                if ub[i] + dis_bound < ub_allowable[i]:
                    ub[i] = ub[i] + dis_bound
                else:
                    ub[i] = ub_allowable[i]

        else:
            for i in range(n):
                lb[i] = lb[i] - dis_bound
                ub[i] = ub[i] + dis_bound

        # input set to check robustness
        I = Star(lb, ub)
        [robust, time, adv_inputs] = obj.isSafe(I, un_robust_reg, method, n_samples, num_cores)

        if robust == 1:
            print('\nThe network is robust with the disturbance dis_bound = %.5f' % dis_bound)
        elif robust == 0:
            print('\nThe network is not robust with the disturbance dis_bound = %.5f, counter examples are found'% dis_bound)
        elif robust == 2:
            print('\nThe robustness of the network is uncertain with the disturbance dis_bound = %.5f, we cannot find counter examples, you can try again with a larger n_samples' % dis_bound)

        t = toc(start)

        return [robust, t, adv_inputs]

    # find maximum robustness value, i.e., maximum disturbance bound
    # that the network is still robust
    def get_robustness_bound(*args):
        # 1: @input_vec: input point
        # 2: @init_dis_bound: initial disturbance bound
        # 3: @dis_bound_step: a step to increase/decrease disturbance bound
        # 4: @max_steps: maximum number of steps for searching
        # 5: @lb_allowable: allowable lower bound of disturbed input:    lb_allowable(i) <= x'[i]
        # 6: @ub_allowable: allowable upper bound of disturbed output:    ub_allowable(i) >= x'[i]
        # x' is the disturbed vector by disturbance bound, |x' - x| <= dis_bound
        # x'[i] >= lb_allowable, x'[i] <= ub_allowable[i]
        # 7: @method: = 'exact-star' or 'approx-star' or 'approx-zono' or 'abs-dom'
        # 8: @n_samples: number of samples used to find counter examples
        # if using over-approximate reachability analysis methods
        # 9: @numCores:  number of cores used in computation
        #
        # @robustness_bound: robustness bound w.r.t @input_vec
        # @t: computation time

        start = tic()
        if len(args) == 11:
            obj = args[0] # FFNNS object
            input_vec = args[1] # input vec
            init_dis_bound = args[2] # initial disturbance bound for searching
            tolerance = args[3] # tolerance (accuracy) for searching
            max_steps = args[4] # maximum searching steps
            lb_allowable = args[5] # allowable lower bound on disturbed inputs
            ub_allowable = args[6] # allowable upper bound on disturbed inputs
            un_robust_reg = args[7] # un-robust region
            method = args[8] # reachability analysis method
            n_samples = args[9] # number of samples used for finding counter examples
            num_cores = args[10] # number of cores used in computation
        elif len(args) == 9:
            obj = args[0] # FFNNS object
            input_vec = args[1] # input vec
            init_dis_bound = args[2] # initial disturbance bound for searching
            tolerance = args[3] # tolerance (accuracy) for searching
            max_steps = args[4] # maximum searching steps
            lb_allowable = args[5] # allowable lower bound on disturbed inputs
            ub_allowable = args[6] # allowable upper bound on disturbed inputs
            un_robust_reg = args[7] # un-robust region
            method = args[8] # reachability analysis method
            n_samples = 1000 # number of samples used for finding counter examples
            num_cores = 1 # number of cores used in computation
        elif len(args) == 7:
            obj = args[0] # FFNNS object
            input_vec = args[1] # input vec
            init_dis_bound = args[2] # initial disturbance bound for searching
            tolerance = args[3] # tolerance (accuracy) for searching
            max_steps = args[4] # maximum searching steps
            lb_allowable = np.array([]) # allowable lower bound on disturbed inputs
            ub_allowable = np.array([]) # allowable upper bound on disturbed inputs
            un_robust_reg = args[7] # un-robust region
            method = args[8] # reachability analysis method
            n_samples = 1000 # number of samples used for finding counter examples
            num_cores = 1 # number of cores used in computation
        elif len(args) == 6:
            obj = args[0] # FFNNS object
            input_vec = args[1] # input vec
            init_dis_bound = args[2] # initial disturbance bound for searching
            tolerance = args[3] # tolerance (accuracy) for searching
            max_steps = args[4] # maximum searching steps
            lb_allowable = np.array([]) # allowable lower bound on disturbed inputs
            ub_allowable = np.array([]) # allowable upper bound on disturbed inputs
            un_robust_reg = args[7] # un-robust region
            method = 'exact-star' # used exact-star reachability analysis method
            n_samples = 1000 # number of samples used for finding counter examples
            num_cores = 1 # number of cores used in computation
        else:
            'error: Invalid number of input arguments (should be 5, 6, 8 or 10)'

        k = 1
        b = init_dis_bound
        bmax = 0
        while k < max_steps:
            print('\nSearching maximum robustness value at step k = %d...' % k)
            print('\nCheck robustness with disturbance bound dis_bound = %.5f...' % b)
            [robust, time, adv_inputs] = obj.isRobust(input_vec, b, un_robust_reg, method, lb_allowable, ub_allowable, n_samples, num_cores)

            if robust == 1:
                bmax = b
                b = b + tolerance
            else:
                b = b - tolerance
                if b == bmax:
                    break
            k = k + 1

        if k == max_steps:
            print('\nCannot find robustness value, increase number of searching steps, i.e., max_steps, and try again')
            robustness_bound = np.array([]);
        else:
            print('\nMaximum robustness value = %.5f is found at k = %d with error tolerance tol = %.5f' % (bmax, k, tolerance))
            robustness_bound = bmax;

        t = toc(start)
        return [robustness_bound, t]

# -------------------------- verify using Deep First Search + exact star for verification (on testing phase)

    # flatten a FFNN into a sequence of operations for reachability
    def flatten(obj, reachMethod):
        # @reachMethod: reachability method

        print('Working on this.....')

    # reachability using flattened network
    def reach_flatten(*args):
        # @inputSet: an array of star set or zonotope
        # @reachMethod: reachability method
        # @numCores: number of cores
        print('Working on this.....')

    def verify_DFS(*args):
        # @inputSets: a star set
        # @unsafeRegion: a HalfSpace object
        # @numCores: number of cores used for verification
        # @safe:  = 'safe' or 'unsafe' or 'unknown'
        # @CEx: counter examples
        print('Working on this.....')

# ---------------------- verify robustness of classification feedforward networks

    def classify(*args):
        # @in_image: a star set or a vector of an image
        # @label_id: output index of classified object

        if len(args) == 2:
            obj = args[0]
            in_image = args[1]
            method = 'approx-star'
            numOfCores = 1
        elif len(args) == 3:
            obj = args[0]
            in_image = args[1]
            method = args[2]
            numOfCores = 1
        elif len(args) == 4:
            obj = args[0]
            in_image = args[1]
            method = args[2]
            numOfCores = args[3]
        else:
            'error: Invalid number of inputs, should be 1, 2, or 3'

        if not isinstance(in_image, Star) and not isinstance(in_image, Zono):
            y = obj.evaluate(in_image)
            label_id = max(y)
        else:
            print('\n=============================================')
            obj.reach(in_image, method, numOfCores)
            RS = obj.outputSet
            n = len(RS)
            label_id = np.array(n, 1)
            for i in range(n):
                label_id[i] = RS[i].getMaxIndexes

        return label_id

    # verify robustness of classification feedforward networks
    def verifyRobustness(*args):
        # @robust: = 1: the network is robust
        #          = 0: the network is notrobust
        #          = 2: robustness is uncertain
        # @counterExamples: a set of counter examples

        if len(args) == 3:
            obj = args[0]
            in_image = args[1]
            correct_id = args[2]
            method = 'approx-star'
            numOfCores = 1
        elif len(args) == 4:
            obj = args[0]
            in_image = args[1]
            correct_id = args[2]
            method = args[3]
            numOfCores = 1
        elif len(args) == 5:
            obj = args[0]
            in_image = args[1]
            correct_id = args[2]
            method = args[3]
            numOfCores = args[4]
        else:
            'error: Invalid number of inputs, should be 2, 3, or 4'

        if correct_id > obj.nO or correct_id < 1:
            'error: Invalid correct id'

        label_id = obj.classify(in_image, method, numOfCores)
        n = len(label_id)
        # check the correctness of classified label
        counterExamples = np.array([])
        incorrect_id = np.array([])
        # should be Image Star
        im1 = in_image.toStar(in_image.dim, 1, 1)

        for i in range(n):
            ids = label_id[i]
            m = len(ids)
            id1 = np.array([])
            for j in range(m):
                if ids[j] != correct_id:
                    id1 = np.column_stack([id1, ids[j]])
            incorrect_id_list = np.column_stack(incorrect_id_list, id1)

            # construct counter example set
            if id1.size:
                if not isinstance(in_image, Star):
                    counterExamples = in_image
                elif method == 'exact-star' and obj.isPieceWiseNetwork:
                    rs = obj.outputSet[i]
                    # should be Image Star
                    rs = rs.toStar(rs.dim, 1, 1)
                    L_len = len(id1)
                    for l in range(L_len):
                        [new_C, new_d] = Star.addConstraint(rs, np.row_stack([1, 1, correct_id]), np.row_stack([1, 1, id1[l]]))
                        counter_IS = Star(im1.V, new_C, im1.pred_lb, im1.pred_ub)
                        counterExamples = np.column_stack([counterExamples, counter_IS.toStar])

        if incorrect_id_list.size:
            robust = 1
            print('\n=============================================')
            print('\nTHE NETWORK IS ROBUST')
            print('\nClassified index: %d' % correct_id)
        else:
            if method == 'exact-star' and obj.isPieceWiseNetwork:
                robust = 0
                print('\n=============================================')
                print('\nTHE NETWORK IS NOT ROBUST')
                print('\nLabel index: %d' % correct_id)
                print('\nClassified index:')

                n = len(incorrect_id_list)
                for i in range(n):
                    print('%d ' % incorrect_id_list[i])
            else:
                robust = 2
                print('\n=============================================')
                print('\nThe robustness of the network is UNCERTAIN due to the conservativeness of approximate analysis')
                print('\nLabel index: %d' % correct_id)
                print('\nPossible classified index: ')

                n = len(incorrect_id_list)
                for i in range(n):
                    print('%d ' % incorrect_id_list[i])

                if obj.isPieceWiseNetwork:
                    print('\nPlease try to verify the robustness with exact-star (exact analysis) option')

        return [robust, counterExamples]

    # verify robustness of classification feedforward networks
    def verifyRBN(*args):
        # @robust: = 1: the network is robust
        #          = 0: the network is notrobust
        #          = 2: robustness is uncertain
        # @cE: a set of counter examples
        # @cands: candidate indexes in the case that the robustness is unknown
        # @vt: verification time

        t = tic()
        if len(args) == 3:
            obj = args[0]
            in_image = args[1]
            correct_id = args[2]
        elif len(args) == 4:
            obj = args[0]
            in_image = args[1]
            correct_id = args[2]
            obj.reachMethod = args[3]
        elif len(args) == 5:
            obj = args[0]
            in_image = args[1]
            correct_id = args[2]
            obj.reachMethod = args[3]
            obj.numCores = args[4]
        elif len(args) == 6:
            obj = args[0]
            in_image = args[1]
            correct_id = args[2]
            obj.reachMethod = args[3]
            obj.numCores = args[4]
            obj.relaxFactor = args[5] # only for the approx-star method
        elif len(args) == 7:
            obj = args[0]
            in_image = args[1]
            correct_id = args[2]
            obj.reachMethod = args[3]
            obj.numCores = args[4]
            obj.relaxFactor = args[5] # only for the approx-star method
            obj.dis_opt = args[6]
        elif len(args) == 8:
            obj = args[0]
            in_image = args[1]
            correct_id = args[2]
            obj.reachMethod = args[3]
            obj.numCores = args[4]
            obj.relaxFactor = args[5] # only for the approx-star method
            obj.dis_opt = args[6]
            obj.lp_solver = args[7]
        elif len(args) == 2:
            obj = args[0]
            # check for struct and field?
        else:
            'error: Invalid number of inputs, should be 2, 3, 4, 5, 6, or 7'

        if correct_id > obj.nO or correct_id < 1:
            'error: Invalid correct id'
        if obj.reachMethod == 'exact-star':
            'error: \nThis method does not support exact-star reachability, please choose approx-star'

        robust = 2 # unknown first
        cands = np.array([])
        cE = np.array([])

        if in_image.state_lb.size:
            y_lb = obj.evaluate(in_image.state_lb)
            max_id = max(y_lb)
            if max_id != correct_id:
                robust = 0
                cE = in_image.state_lb

            y_ub = obj.evaluate(in_image.state_ub)
            max_id = max(y_ub)
            if max_id != correct_id:
                robust = 0
                cE = in_image.state_ub

        if robust == 2:
            obj.reach(in_image, obj.reachMethod, obj.numCores, obj.relaxFactor, obj.dis_opt, obj.lp_solver)
            R = obj.outputSet
            [lb, ub] = R.estimateRanges
            max_val = lb(correct_id)

            # need to correct
            #max_cd = find(ub>max_val) # max point candidates

            flatten_ub = np.ndarray.flatten(ub, 'F')
            map = np.argwhere(flatten_ub > max_val)
            max_cd = np.array([])
            for i in range(len(map)):
                # index = map[i][1] * len(flatten_ub[0]) + map[0][i]
                index = map[i][1]
                print(index)
                max_cd = np.append(max_cd, index)

            max_cd[max_cd == correct_id] = np.array([]) # delete the max_id

            if max_cd.size:
                robust = 1
            else:
                n = len(max_cd)
                count = 0
                for i in range(n):
                    # need to correct
                    if R.is_p1_larger_than_p2(max_cd[i], correct_id):
                        cands = max_cd[i]
                        break
                    else:
                        count = count + 1

                    if count == n:
                        robust = 1

        vt = toc(t)
        return [robust, cE, cands, vt]

    # evaluate robustness of a classification feedforward network on an array of input (test) sets
    def evaluateRBN(*args):
        # @in_images: input sets
        # @correct_ids: an array of correct labels corresponding to the input sets
        # @method: reachability method: 'exact-star', 'approx-star',
        # 'approx-zono' and 'abs-dom'
        # @numCores: number of cores used in computation
        # @r: robustness value (in percentage)
        # @rb: robustness results
        # @cE: counterexamples
        # @cands: candidate idexes
        # @vt: verification times

        if len(args) == 8:
            obj = args[0]
            in_images = args[1]
            correct_ids = args[2]
            obj.reachMethod = args[3]
            obj.numCores = args[4]
            obj.relaxFactor = args[5] # only for the approx-star method
            obj.dis_opt = args[6]
            obj.lp_solver = args[7]
        elif len(args) == 7:
            obj = args[0]
            in_images = args[1]
            correct_ids = args[2]
            obj.reachMethod = args[3]
            obj.numCores = args[4]
            obj.relaxFactor = args[5] # only for the approx-star method
        elif len(args) == 5:
            obj = args[0]
            in_images = args[1]
            correct_ids = args[2]
            obj.reachMethod = args[3]
            obj.numCores = args[4]
        elif len(args) == 4:
            obj = args[0]
            in_images = args[1]
            correct_ids = args[2]
            obj.reachMethod = args[3]
            obj.numCores = 1
        elif len(args) == 2:
            obj = args[0]
        else:
            'error: Invalid number of input arguments, should be 2, 3, 4, 5, 6, or 7'

        N = len(in_images)
        if len(correct_ids) != N:
            'error: Inconsistency between the number of correct_ids and the number of input sets'

        count = np.zeros(1, N)
        rb = np.zeros(1, N)
        cE = np.array(1, N)
        cands = np.array(1, N)
        vt = np.zeros(1, N)

        if obj.reachMethod != 'exact-star':
            # verify reachable set
            if obj.numCores > 1 and N > 1:
                print('working on this.....')
            else:
                for i in range(N):
                    [rb[i], cE[i], cands[i], vt[i]] = obj.verifyRBN(in_images[i], correct_ids[i], obj.reachMethod, obj.numCores, obj.relaxFactor, obj.dis_opt, obj.lp_solver )
                    if rb[i] == 1:
                        count[i] = 1
                    else:
                        count[i] = 0
        r = sum(count)/N
        return [r, rb, cE, cands, vt]

    #  evaluate robustness of a classification feedforward network on an array of input (test) sets
    def evaluateRobustness(*args):
        # @in_images: input sets
        # @correct_ids: an array of correct labels corresponding to the input sets
        # @method: reachability method: 'exact-star', 'approx-star',
        # 'approx-zono' and 'abs-dom'
        # @numCores: number of cores used in computation
        # @r: robustness value (in percentage)

        if len(args) == 5:
            obj = args[0]
            in_images = args[1]
            correct_ids = args[2]
            method = args[3]
            numOfCores = args[4]
        elif len(args) == 4:
            obj = args[0]
            in_images = args[1]
            correct_ids = args[2]
            method = args[3]
            numOfCores = 1
        else:
            'error: Invalid number of input arguments, should be 3 or 4'

        N = len(in_images)
        if len(correct_ids) != N:
            'error: Inconsistency between the number of correct_ids and the number of input sets'

        count = np.zeros(1, N)
        if method == 'exact-star':
            # verify reachable set
            if numOfCores > 1:
                print('working on this...')
            else:
                for i in range(N):
                    [rb, counterExamples] = obj.verifyRobustness(in_images[i], correct_ids[i], method)
                    if rb == 1:
                        count[i] = 1
                    else:
                        count[i] = 0

        r = sum(count)/N
        return r

# --------------------- parse Matlab/python trained Feedforward network
    def parse(*args):
        # @MatlabNet: A feedforward network trained by Matlab
        # @name: name of the network
        # @nnvNet: an NNV FFNNS object for reachability analysis and
        # verification

        print('Working on this....')

    # random generate a network for testing
    def rand(neurons, funcs):
        # neurons: an array of neurons of input layer - hidden layers-
        # output layers
        # funcs: an array of activation functions of hidden layers

        print('Working on this....')

# -------------------- method for measuring the influence of neurons in the network to the output sets
    def getInfluence(*args):
        # del: value of perturbance
        # nCore: number of cores used for computation
        # @infl: sorted influence vectors for all neurons in the network
        # @ids: sorted neuron indexes in descend manner, i.e., from
        # high influence to low influence

        if len(args) == 2:
            obj = args[0]
            del_val = args[1]
            nCores = 1
        elif len(args) == 3:
            obj = args[0]
            del_val = args[1]
            nCores = args[2]
        else:
            'error: Invalid number of input arguments'

        infl = np.array(1, obj.nL - 1)
        ids = np.array(1, obj.nL - 1)

        for i in range(obj.nL - 1):
            [infl[i+1], ids[i+1]] = obj.getLayerInfluence(i, del_val, nCores)

        return [infl, ids]

    # # get influence measurement for a single layer
    # # measure how a single neuron in a layer affects to the size of the
    # # output set,
    # def getLayerInfluence(*args):
    #     # @layer_id: layer index
    #     # @del: perturbance to a neuron |x(i) - x'(i)| <= del
    #     # @nCores: number of cores used for computation
    #     # @infl: sorted influence measurement vector
    #     # @ids: sorted neurons indexes in descend manner, i.e., from high influence to
    #     # low-influence
    #
    #     if len(args) == 3:
    #         obj = args[0]
    #         layer_id = args[1]
    #         del_val = args[2]
    #         nCores = 1
    #     elif len(args) == 4:
    #         obj = args[0]
    #         layer_id = args[1]
    #         del_val = args[2]
    #         nCores = args[3]
    #     else:
    #         'error: Invalid number of input arguments'
    #
    #     if layer_id < 0 and layer_id > obj.nL - 1:
    #         'error: Invalid layer index'
    #     if nCores < 1:
    #         'error: Invalid number of cores'
    #     if del_val < 0:
    #         'error: Invalid value of perturbance, should be larger than '
    #
    #     # construct a subnetwork
    #     # need to fix
    #     subnet = FFNNS(obj.Layers(layer_id + 1:obj.nL))
    #     N = subnet.nI
    #     lb = np.zeros(N, 1)
    #     influence = np.zeros(N, 1)
    #
    #     if nCores > 1:
    #         print('Working on this....')
    #     else:
    #         for i in range(N):
    #             ub = np.zeros(N, 1)
    #             ub[i] = del_val
    #             I = Box(lb, ub)
    #             I = I.toZono() # input set to the subnet
    #             [O, time] = subnet.reach(I, 'approx-zono')
    #             [lb1, ub1] = O.getBounds
    #             influence[i] = max(ub1-lb1)/del_val
    #
    #     # need to fix
    #     [infl, ids] = sorted(influence)
    #
    #     return [infl, ids]

    # def sub_verify_DFS(ops, inputSet, start_index, unsafeRegion, reachMethod):
    #     # @ops: operations array
    #     # @inputSet: a star set
    #     # @start_index: the index of the sup-tree to start searching
    #     # @unsafeRegion: a HalfSpace
    #     # @reachMethod: reachability method
    #
    #     N = len(ops)
    #     U = unsafeRegion
    #     S.data = inputSet
    #     S.opsIndex = start_index
    #     safe = 'safe'
    #     CEx = np.array([])
    #     numSets = 0
    #     print('\n Number of verified reach sets: 0000000000')
    #     while safe == 'safe' and S.size:
    #         S1 = S[1].data
    #         id = S[1].opsIndex
    #         if id < N:
    #             S2 = ops[id].execute(S1)
    #             if len(S2) == 2:
    #                 S3_1.data = S2[1]
    #                 S3_1.opsIndex = id + 1
    #                 S3_2.data = S2[2]
    #                 S3_2.opsIndex = id + 1
    #                 S3 = np.column_stack([S3_1, S3_2])
    #                 S[1] = np.array([])
    #                 S = np.column_stack([S3, S])
    #             else:
    #                 S4.data = S2
    #                 S4.opsIndex = id + 1
    #                 S[1] = np.array([])
    #                 S = np.column_stack([S4, S])
    #         else:
    #             # checking safety of the leaf sets
    #             S2 = ops[id].execute(S1)
    #             if len(S2) == 2:
    #                 H1 = S2[1].intersectHalfSpace(U.G, U.g)
    #                 H2 = S2[2].intersectHalfSpace(U.G, U.g)
    #                 if H1.size:
    #                     if reachMethod == 'exact-star':
    #                         safe = 'unsafe'
    #                         CEx = Star(inputSet.V, H1.C, H1.d, H1.predicate_lb, H1.predicate_ub)
    #                     else:
    #                         safe = 'unknown'
    #                         CEx = np.array([])
    #                 elif H2.size:
    #                     if reachMethod == 'exact-star':
    #                         safe = 'unsafe'
    #                         CEx = Star(inputSet.V, H2.C, H2.d, H2.predicate_lb, H2.predicate_ub)
    #                     else:
    #                         safe = 'unknown'
    #                         CEx = np.array([])
    #             else:
    #                 H = S2.intersectHalfSpace(U.G, U.g)
    #                 if H.size:
    #                     if reachMethod == 'exact-star':
    #                         safe = 'unsafe'
    #                         CEx = Star(inputSet.V, H.C, H.d, H.predicate_lb, H.predicate_ub)
    #                     else:
    #                         safe = 'unknown'
    #                         CEx = np.array([])
    #
    #
    #         S[1] = np.array([])
    #         numSets = numSets + 1
    #         print("\b\b\b\b\b\b\b\b\b\b%10d" % numSets)
    #     print('\n')
    #     return [safe, CEx]

