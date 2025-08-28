# -*- coding: utf-8 -*-
"""
Contans graphing functions for;
    xn vs xn+1 for Logistic and Digital Map
    Cobweb plot of Logistic and Digital Map

@author: Marius Furtig-Rytterager
"""

import numpy as np
import matplotlib.pyplot as plt
from Digital_Map import digital_map

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

def graph_logistic_map(x, n, save = False):
    '''

    Parameters
    ----------
    x : Seed/Initial Input
    
    n : Number of iterations
    
    save : When true, the resulting graph is saved to a svg file.
           Defaults to being false

    Returns
    -------
    Graph of xn vs xn+1 for Logistic Map

    '''
    
    # Generating Sequence of Random Numbers for each Map
    x_vals = Logistic_map(x, n)  
    
    # Generating Sequence without last entry
    x_n = x_vals[:-1] 
    
    # Generating Sequence without last entry
    x_np1 = x_vals[1:] 

    # Plotting Graph  
    
    fig, _ = plt.subplots(figsize=(6, 6))
    plt.plot(x_n, x_np1, 'bo', label = 'Logistic Map')
    plt.legend()
    plt.grid()
    plt.xlabel('Xn')
    plt.ylabel('Xn+1')
    plt.title('Plot of Next State vs Current State for the Logistic Map')
    
    # Add Description
    fig.subplots_adjust(bottom=0.15)
    fig.text(0.5, 0.05, f"Shows the dynamics of the Logistic Map with r = 4 and x = {x}",
         ha='center', fontsize=10)
    
    # Save Plot if Enabled
    if save == True:
        plt.savefig(f"Logistic_Map_xn_vs_xn+1_{x}_{n}.svg") 
        
    plt.show() 

def graph_digital_map(x, n, k, save = False):
    '''

    Parameters
    ----------
    x : Seed/Initial Input
    
    n : Number of iterations
    
    k : Number of digits in each binary number
    
    save : When true, the resulting graph is saved to a svg file.
           Defaults to being false

    Returns
    -------
    Graph of xn vs xn+1 for Digital Map

    '''
    
    # Generating Sequence of Random Numbers for each Map 
    x_vals_digital, _ = digital_map(x, n, k) 
    
    # Generating Sequence without last entry
    x_n_digital = x_vals_digital[:-1] 
    
    # Generating Sequence without last entry
    x_np1_digital = x_vals_digital[1:] 

    # Plotting Graph  
    
    fig, _ = plt.subplots(figsize=(6, 6))
    plt.plot(x_n_digital, x_np1_digital, 'go', label = 'Digital Map')
    plt.legend()
    plt.grid()
    plt.xlabel('Xn')
    plt.ylabel('Xn+1')
    plt.title('Plot of Next State vs Current State for the Digital Map')
        
    # Add Description
    fig.subplots_adjust(bottom=0.15)
    fig.text(0.5, 0.05, f"Shows the dynamics of the Digital Map with r = 4, x = {x} and k = {k}",
         ha='center', fontsize=10)
    
    # Save Plot if Enabled
    if save == True:
        plt.savefig(f"Digital_Map_xn_vs_xn+1_{x}_{n}_{k}.svg") 
        
    plt.show() 

def graph_all_maps(x, n, k, save = False):
    '''

    Parameters
    ----------
    x : Seed/Initial Input
    
    n : Number of iterations
    
    k : Number of digits in each binary number
    
    save : When true, the resulting graph is saved to a svg file.
           Defaults to being false

    Returns
    -------
    xn vs xn+1 for Digital and Logistic Map on same graph

    '''
    
    # Generating Sequence of Random Numbers for each Map 
    x_vals_digital, _ = digital_map(x, n, k) 
    x_vals = Logistic_map(x, n)  
    
    # Generating Sequence without last entry
    x_n_digital = x_vals_digital[:-1] 
    x_n = x_vals[:-1] 
    
    # Generating Sequence without last entry
    x_np1_digital = x_vals_digital[1:] 
    x_np1 = x_vals[1:] 

    # Plotting Graph  
    
    fig, _ = plt.subplots(figsize=(6, 6))
    plt.plot(x_n_digital, x_np1_digital, 'go', label = 'Digital Map')
    plt.plot(x_n, x_np1, 'bo', label = 'Logistic Map')
    plt.legend()
    plt.grid()
    plt.xlabel('Xn')
    plt.ylabel('Xn+1')
    plt.title('Plot of Next State vs Current State for the Logistic Map')
        
    # Add Description
    fig.subplots_adjust(bottom=0.15)
    fig.text(0.5, 0.05, f"Shows the dynamics of the Logistic and Digital Map with r = 4, x = {x} and k = {k}",
         ha='center', fontsize=10)
    
    # Save Plot if enabled
    if save == True:
        plt.savefig(f"Digital_and_Logistic_Map_xn_vs_xn+1_{x}_{n}_{k}.svg") 
        
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
    
    save : When true, the resulting graph is saved to a svg file. 
           Defaults to being false

    Returns
    -------
    Graphs cobweb plot for Logistic Map

    '''
    # Generate x values and f(x)
    x = np.linspace(0.001, 0.999, 500)
    fx_total = np.array([one_step_map_logistic(xi) for xi in x])

    fig, _ = plt.subplots(figsize=(6, 6))
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
    
    # Graph Cobweb Plot
    plt.plot(x_vals, fx_vals, color='blue', lw=1, label='Cobweb Path')
    plt.title('Cobweb Plot of Logistic Map')
    plt.xlabel('$x_n$')
    plt.ylabel('$x_{n+1}$')
    plt.legend()
    plt.grid(True)
    
    # Add Description
    fig.subplots_adjust(bottom=0.15)
    fig.text(0.5, 0.05, f"Displays the cobweb plot of the Logistic Map with r = 4 and x = {x0}",
         ha='center', fontsize=10)
    
    # Save Plot if enabled
    if save == True:
        plt.savefig(f"Logistic_Cobweb_Plot_of_{x0}_{n}.svg") 
        
    plt.show()

def cobweb_plot_digital(x0, n, k, save = False):
    '''

    Parameters
    ----------
    x0 : Seed/Initial Input
        
    n : Number of iterations
    
    k : Number of digits in each binary number
    
    save : When true, the resulting graph is saved to a svg file. 
           Defaults to being false 

    Returns
    -------
    Graphs cobweb plot for Digital Map

    '''   
    # Generate x values and f(x)
    x = np.linspace(0.001, 0.999, 500)
    fx_total = np.array([one_step_map_digital(xi, k) for xi in x])

    fig, _ = plt.subplots(figsize=(6, 6))
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
    
    # Graph Cobweb Plot
    plt.plot(x_vals, fx_vals, color='green', lw=1, label='Cobweb Path')
    plt.title('Cobweb Plot of Digital Map')
    plt.xlabel('$x_n$')
    plt.ylabel('$x_{n+1}$')
    plt.legend()
    plt.grid(True)
    
    # Add Description
    fig.subplots_adjust(bottom=0.15)
    fig.text(0.5, 0.05, f"Displays the cobweb plot of the Digital Map with r = 4, x = {x0} and k = {k}",
         ha='center', fontsize=10)
    
    # Save Plot if enabled
    if save == True:
        plt.savefig(f"Digital_Cobweb_Plot_of_{x0}_{n}.svg") 
        
    plt.show()

X = np.array([0.324, 0.617, 0.888])
k = 64

for i in range(len(X)):
    cobweb_plot_logistic(x0 = X[i], n = 100, save = True)
    cobweb_plot_logistic(x0 = X[i], n = 1000, save = True)
    cobweb_plot_digital(x0 = X[i], n = 100, k = k, save = True)
    cobweb_plot_digital(x0 = X[i], n = 1000, k = k, save = True)