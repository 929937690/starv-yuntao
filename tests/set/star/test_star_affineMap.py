#!/usr/bin/python3
"""
Created on Tue Oct  5 16:35:49 2021

@author: Apala
"""

import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.star import Star


def main():
    
    lb = np.matrix('1;1')
    ub = np.matrix('2;2')
    
   

    S = Star(lb=lb, ub=ub)
    print(S)
     
    V = S.V
    C = S.C
    d = S.d
    

    S2 = Star(V, C, d)
    print(S2)
    
    w = np.matrix('1 -1; 1 1')
    b = np.matrix('0.5;0.5')
    
    
    z = S2.affineMap(w,b)
    print(z)    
 
    
if __name__ == '__main__':
    main()
