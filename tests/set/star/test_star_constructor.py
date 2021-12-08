#!/usr/bin/python3
"""
Created on Tue Oct  5 14:18:10 2021

@author: Apala
"""

import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.star import Star
from engine.set.box import Box

def main():
    
    lb = np.matrix('1;1')
    ub = np.matrix('2;2')
    
    S = Star(lb=lb, ub=ub)
    print(S.__repr__())
    
    V = S.V
    C = S.C
    d = S.d
    pred_lb = S.predicate_lb
    pred_ub = S.predicate_ub

    S2 = Star(V, C, d, pred_lb=pred_lb, pred_ub=pred_ub)
    print(S2.__repr__())

    
if __name__ == '__main__':
    main()