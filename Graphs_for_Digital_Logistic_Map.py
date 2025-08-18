# -*- coding: utf-8 -*-
"""
Shows Graphed data for Digital logistic Map

@author: 22391643
"""

import numpy as np
import matplotlib.pyplot as plt
from Digital_Logistic_Map import digital_logistic_map

x = 0.945 # Starting Condition
n = 100 # Number of Iterations
k = 64 # Number of digits

# Logistic Map

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

# Graphing Accuracy 
Accuracy = np.arange(0, n+1, dtype = float)
Range = np.arange(0, n+1)
Digital_vals, _ = digital_logistic_map(x, n, k)
Logistic_vals = Logistic_map(x, n)

for t in range(len(Accuracy)):
    Accuracy[t] = Digital_vals[t] - Logistic_vals[t]
    
# Plotting Graph
plt.plot(Range, Accuracy, 'ro', label = 'Accuracy')
plt.legend()
plt.grid()
plt.xlabel('Range')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.show()   

# Graphing Xn vs Xn+1 for all Logistic Map 

x_vals = Logistic_map(x, n)  # one long trajectory
x_vals_digital, _ = digital_logistic_map(x, n, k)

# Plotting Graph    

x_n = x_vals[:-1] # all but last
x_n_digital = x_vals_digital[:-1] # all but last

x_np1 = x_vals[1:] # all but first
x_np1_digital = x_vals_digital[1:] # all but first

plt.plot(x_n, x_np1, 'ro', label = 'logistic map')
plt.plot(x_n_digital, x_np1_digital, 'go', label = 'digital map')
plt.legend()
plt.grid()
plt.xlabel('Xn')
plt.ylabel('Xn+1')
plt.title('Digital Logistic Map - Xn+1 vs Xn')
#plt.savefig('Digital Logistic Map - Xn+1 vs Xn.png') 
plt.show()   

def one_step_map(x):
    fmap = Logistic_map(x, 1)[-1] # take the 2nd value: f(x)
    return fmap  

def one_step_map_for_digital(x):
    xN,_ = digital_logistic_map(x, 1, k)  
    fmap = xN[-1] # take the 2nd value: f(x)
    return fmap

def cobweb_plot(f, x0, n):
    '''
    f : map function that returns f(x)
    x0 : initial condition
    k  : number of steps
    '''
    # Generate x values and f(x)
    x = np.linspace(0.001, 0.999, 500)
    y = np.array([f(xi) for xi in x])

    plt.figure(figsize=(8, 8))
    plt.plot(x, y, label='f(x)')
    plt.plot(x, x, 'k--', label='y = x')  # Diagonal

    # Initialize cobweb path
    x_vals = [x0]
    y_vals = [0]  # starts from (x0, 0)
    for _ in range(n):
        y = f(x_vals[-1])
        x_vals.append(x_vals[-1])  # vertical up to f(x)
        y_vals.append(y)
        x_vals.append(y)           # horizontal to (f(x), f(x))
        y_vals.append(y)

    plt.plot(x_vals, y_vals, color='red', lw=1, label='Cobweb Path')

    plt.title(f"Cobweb Plot of {f} for n = {n} and x0 = {x0}")
    plt.xlabel('$x_n$')
    plt.ylabel('$x_{n+1}$')
    plt.legend()
    plt.grid(True)
    #plt.savefig(f"Cobweb Plot of {f}.png") 
    plt.show()

# Usage

cobweb_plot(one_step_map, x0=x, n = 100)
cobweb_plot(one_step_map_for_digital, x0=x, n= 100)



