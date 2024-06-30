import numpy as np
import torch as pt
import scipy
from scipy import stats
import sympy as sp
from IPython.display import display

x      =  sp.symbols('x')

def discretize_axis(xmin, xmax, xbins):
    """
    xcenters, xedges, xbins, dx = discretize_axis(xmin, xmax, xbins)
    """

    xedges   = np.linspace(xmin, xmax, xbins) 
    dx       = xedges[1] - xedges[0]
    xcenters = xedges + 0.5 * dx
    xcenters = np.delete(xcenters,-1)
    xbins    = len(xcenters)

    return xcenters, xedges, xbins, dx


class Potential1D_car:
    def __init__(self, 
                 V   = (x**2 - 1) ** 2 
                ):
        
        """
        ... 
        """    
        display(V)
        dVx = V.diff(x)
        
        self.V    = sp.lambdify((x), V,   modules=['numpy'])
        self.dVx  = sp.lambdify((x), dVx, modules=['numpy'])


class SystemParameters:
    def __init__(self, 
                 Ndims  = 1, 
                 mass   = np.array([1.0]), 
                 gamma  = np.array([1.0]), 
                 kB     = 0.008314463, # kJ mol^-1 K^-1
                 T      = 300 # K              
                ):
        
        """
        ... 
        """        
        
        self.Ndims = Ndims
        self.mass  = mass
        self.gamma = gamma
        self.kB    = kB
        self.T     = T
        self.beta  = 1 / kB / T # kJ^-1
        
def EM_integrator_1D(gradV, ):
    xt = np.zeros((Npoints, Nfinpoints))
    
    for n in tqdm(range(Npoints)):
    
        x  = x0[n] * np.ones(Nfinpoints)
        
        for k in range(Nsteps2-1):
    
            eta     =  np.random.normal(0, 1, Nfinpoints)
            force   =  - gradV(x)
            x       =  x + 1 / mass / gamma * force * dt + sigma * eta * sdt
                        
        xt[n,:] = x
    

def scale_and_shift(y):
    """
    scale and shift function for the modified power method
    """
    minarr = np.min(y)
    maxarr = np.max(y)
    hat_y  = (y - minarr) / (maxarr - minarr)

    return hat_y

def exit_rates_from_chi(tau, chi_0, chi_tau):
    
    #
    chi1      = chi_0[:,0]
    chi2      = chi_0[:,1]

    #
    prop_chi1 = chi_tau[:,0]
    prop_chi2 = chi_tau[:,1]

    res1 = scipy.stats.linregress(chi1, prop_chi1)
    res2 = scipy.stats.linregress(chi2, prop_chi2)

    rate1  = - 1 / tau * np.log( res1.slope ) * ( 1 + res1.intercept  / ( res1.slope - 1 ))
    rate2  = - 1 / tau * np.log( res2.slope ) * ( 1 + res2.intercept  / ( res2.slope - 1 ))
    
    #

    print('Exit rate 1:', rate1)
    print('Exit rate 2:', rate2)

    return rate1, rate2

    



