# -*- coding: utf-8 -*-
"""
Gives Logistic Map

@author: 22391643
"""
import numpy as np

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

def transform_logistic_map(xN, T, d):
    
    def inverse_transform(x):
        
        y = (1/np.pi)*(np.arcsin(2*x - 1))
        return y
    
    xN = xN[int(T):]
    yN = np.zeros(len(xN), dtype = np.longdouble)
    for i in range (len(xN)):
        yN[i] = inverse_transform(xN[i]) + 1/2
        
    yN = yN[::d]
    
    return yN
        
 


