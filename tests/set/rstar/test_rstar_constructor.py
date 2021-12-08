#!/usr/bin/env python3
import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.box import Box

from engine.set.rstar import RStar

def main():
    lb = np.matrix('-1; -1')
    ub = np.matrix('1; 1')

    RS = RStar(lb = lb, ub = ub)
    print(RS.__repr__())
    print(RS)

if __name__ == '__main__':
    main()