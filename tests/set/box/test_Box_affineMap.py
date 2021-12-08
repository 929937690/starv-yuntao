#!/usr/bin/env python3
import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.box import Box

def main():
    lb = np.matrix('-1; -1; -1')
    ub = np.matrix('1; 1; 1')

    B = Box(lb,ub)
    print(B.__repr__())

    W = np.matrix(np.random.rand(3,3))
    b = np.matrix(np.random.rand(3,1))

    print('W:\n%s\n' % W)
    print('b:\n%s\n' % b)

    B = B.affineMap(W,b)
    print(B.__repr__())
    

if __name__ == '__main__':
    main()