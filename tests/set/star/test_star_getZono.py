#!/usr/bin/python3
"""
Created on Wed Oct  6 17:07:47 2021

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
    
    lb = np.matrix('-1;1')
    ub = np.matrix('-2;2')  
   

    S = Star(lb=lb,ub=ub)
    
    Z = S.getZono()
    print(Z)
    
    
    
if __name__ == '__main__':
    main()    