# -*- coding: utf-8 -*-
"""
Gives Logistic Map

@author: 22391643
"""
import numpy as np
from numba import njit, prange

def Logistic_map(x, n):
    
    #catching errors
    if x<=0 or x>=1:
        print('Error')
    else:
        X=np.zeros(n+1)
        X[0]=x
        for j in range(n):
             x=4.0*x*(1.0-x)
             X[j+1]=x
             
        return X

@njit(parallel = True)
def transform_logistic_map(xN, T, d):
    
    def inverse_transform(x):
        
        y = (1/np.pi)*(np.arcsin(2*x - 1))
        return y
    
    xN = xN[int(T):]
    yN = np.zeros(len(xN), dtype = np.float64)
    
    for i in prange (len(xN)):
        
        #Catching rounding errors
        t = np.float64(2.0)*xN[i] - np.float64(1.0)
        if t > 1.0: t = 1.0
        if t < -1.0: t = -1.0
        
        yN[i] = inverse_transform(xN[i]) + np.float64(0.5)
        
    yN = yN[::d]
    
    return yN




