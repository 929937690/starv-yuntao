#!/usr/bin/python3
import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.zono import Zono

def main():
    c1 = np.matrix('0; 0')
    V1 = np.matrix('1 -1; 1 1; 0.5 0; -1 0.5')
    Z1 = Zono(c1, V1.transpose())
    I1 = Z1.toStar()
    
    I2 = I1.getOrientedBox()
    print(I2.__repr__)
    I2.getRanges()
    I3 = I1.getBox()
    print(I3.getRange())
    
if __name__ == '__main__':
    main()