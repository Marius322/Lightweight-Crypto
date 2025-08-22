# -*- coding: utf-8 -*-
"""
Contans graphing functions for;
    xn vs xn+1 for Logistic and Digital Map
    Cobweb plot of Logistic and Digital Map

@author: Marius Furtig-Rytterager
"""

import numpy as np
import matplotlib.pyplot as plt
from Digital_Logistic_Map import digital_map

# Logistic Map

def Logistic_map(x, n):
    '''
    This is the real-valued logistic map

    Parameters
    ----------
    x : Seed/Initial input
        
    n : Number of iterations

    Returns
    -------
    X : A NumPy array of random numbers, determined by Logistic Map

    '''  
    # Catching Errors
    if x <= 0 or x >= 1:
        raise ValueError('x must lie between 0 and 1')
        
    if not isinstance(n, int) or n <= 0:
       raise ValueError('n must an be integer greater than 0')
       
    X = np.zeros(n+1)
    X[0] = x
    for j in range(n):
        x = 4.0 * x * (1.0-x)
        X[j+1] = x
             
    return X

# Graphing Xn vs Xn+1 for Logistic and Digital Map 

def Graph_Map(x, n, k, save = False):
    '''

    Parameters
    ----------
    x : Seed/Initial Input
    
    n : Number of iterations
    
    k : Number of digits in each binary number
    
    save : When true, the resulting graph is saved to a png file.
           Defaults to being false

    Returns
    -------
    Graph of xn vs xn+1 for Logistic and Digital Map

    '''
    
    # Generating Sequence of Random Numbers for each Map
    x_vals = Logistic_map(x, n)  
    x_vals_digital, _ = digital_map(x, n, k) 
    
    # Generating Sequence without last entry
    x_n = x_vals[:-1] 
    x_n_digital = x_vals_digital[:-1] 
    
    # Generating Sequence without last entry
    x_np1 = x_vals[1:] 
    x_np1_digital = x_vals_digital[1:] 

    # Plotting Graph  
    plt.plot(x_n, x_np1, 'ro', label = 'logistic map')
    plt.plot(x_n_digital, x_np1_digital, 'go', label = 'digital map')
    plt.legend()
    plt.grid()
    plt.xlabel('Xn')
    plt.ylabel('Xn+1')
    plt.title('Digital Logistic Map - Xn+1 vs Xn')
    
    if save == True:
        plt.savefig(f"Digital Logistic Map - Xn+1 vs Xn - {x}_{n}.png") 
        
    plt.show() 

# Helper Functions

def one_step_map_logistic(x):
    '''
    Helper function for cobweb plot

    Parameters
    ----------
    x : Seed/Initial Input

    Returns
    -------
    second_value : Returns the next entry in the Logistic Map

    '''
    second_value = Logistic_map(x, 1)[-1] 
    return second_value  

def one_step_map_digital(x, k):
    '''   
    Helper function for cobweb plot

    Parameters
    ----------
    x : Seed/Initial Input
    
    k : Number of digits of each binary number

    Returns
    -------
    second_value : Returns the next entry in the Digital Map

    '''
    xN,_ = digital_map(x, 1, k)  
    second_value = xN[-1] 
    return second_value

# Cobweb Plot for Logistic and Digital Maps

def cobweb_plot_logistic(x0, n, save = False):
    '''

    Parameters
    ----------
    x0 : Seed/Initial Input
        
    n : Number of iterations
    
    save : When true, the resulting graph is saved to a png file. 
           Defaults to being false

    Returns
    -------
    Graphs cobweb plot for Logistic Map

    '''
    # Generate x values and f(x)
    x = np.linspace(0.001, 0.999, 500)
    fx_total = np.array([one_step_map_logistic(xi) for xi in x])

    plt.figure(figsize=(8, 8))
    plt.plot(x, fx_total, label='f(x)') # Graph of Logistic Map
    plt.plot(x, x, 'k--', label='y = x')  # Diagonal Line

    # Initialize cobweb path, starting from (x0, 0)
    x_vals = [x0]
    fx_vals = [0]  
    
    for _ in range(n):
        
        fx = one_step_map_logistic(x_vals[-1])
        
        # Draws vertical line from (x, f(previous x)) to (x, f(x))
        x_vals.append(x_vals[-1])  
        fx_vals.append(fx)
        
        # Draws horizontal line from (x, f(x)) to (f(x), f(x))
        x_vals.append(fx) 
        fx_vals.append(fx)

    plt.plot(x_vals, fx_vals, color='red', lw=1, label='Cobweb Path')
    plt.title('Cobweb Plot of Logistic Map')
    plt.xlabel('$x_n$')
    plt.ylabel('$x_{n+1}$')
    plt.legend()
    plt.grid(True)
    
    if save == True:
        plt.savefig(f"Logistic Cobweb Plot of {x0}_{n}.png") 
        
    plt.show()

def cobweb_plot_digital(x0, n, k, save = False):
    '''

    Parameters
    ----------
    x0 : Seed/Initial Input
        
    n : Number of iterations
    
    k : Number of digits in each binary number
    
    save : When true, the resulting graph is saved to a png file. 
           Defaults to being false 

    Returns
    -------
    Graphs cobweb plot for Digital Map

    '''   
    # Generate x values and f(x)
    x = np.linspace(0.001, 0.999, 500)
    fx_total = np.array([one_step_map_digital(xi, k) for xi in x])

    plt.figure(figsize=(8, 8))
    plt.plot(x, fx_total, label='f(x)') # Graph of Digital Map
    plt.plot(x, x, 'k--', label='y = x')  # Diagonal Line

    # Initialize cobweb path, starting from (x0, 0)
    x_vals = [x0]
    fx_vals = [0]  
    
    for _ in range(n):
        
        fx = one_step_map_digital(x_vals[-1], k)
        
        # Draws vertical line from (x, f(previous x)) to (x, f(x))
        x_vals.append(x_vals[-1])  
        fx_vals.append(fx)
        
        # Draws horizontal line from (x, f(x)) to (f(x), f(x))
        x_vals.append(fx) 
        fx_vals.append(fx)

    plt.plot(x_vals, fx_vals, color='red', lw=1, label='Cobweb Path')
    plt.title('Cobweb Plot of Digital Map')
    plt.xlabel('$x_n$')
    plt.ylabel('$x_{n+1}$')
    plt.legend()
    plt.grid(True)
    
    if save == True:
        plt.savefig(f"Digital Cobweb Plot of {x0}_{n}.png") 
        
    plt.show()

