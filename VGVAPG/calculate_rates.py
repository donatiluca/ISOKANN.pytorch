print(" ")
print(" S T A R T ")
print(" ")

# directories
main_dir =  '../../VGVAPG/implicit_0/'
inp_dir  =  main_dir + 'input/'
out_dir  =  main_dir + 'output/'

import mdtraj as md
import numpy as np
import sympy as sp
import torch as pt
from tqdm import tqdm
import scipy
import itertools
import matplotlib.pyplot as plt
from functionsNN import NeuralNetwork, trainNN, exit_rates_from_chi

if pt.cuda.is_available():
    dev = pt.device('cuda')
else:
    dev = pt.device('cpu')



font = {'size'   : 10}
plt.rc('font', **font)
in2cm = 1/2.54  # centimeters in inches



# Load initial states
D0 = pt.load(main_dir + 'D0.pt', map_location=dev)
Npoints = D0.shape[0]
Ndims   = D0.shape[1]
print('Shape of D0?')
print('Npoints, Ndims')
print(D0.shape)
print(" ")

# Load final states
DT = pt.load(main_dir + 'Dt.pt', map_location=dev)
Nfinpoints  = DT.shape[1]

# Select only final states between 0 and 10
frame = 9
Dt = DT[:,0:10,:,frame]
print('Shape of Dt?')
print('Npoints, Nfinpoints, Ndims')
print(Dt.shape)

nodes = np.loadtxt('nodes.txt').astype('int')
f_NN = NeuralNetwork( Nodes = nodes, enforce_positive = 0 ).to(dev)
f_NN.load_state_dict(pt.load('f_NN.pt'))

chi_0 = f_NN(D0).cpu().detach().numpy()


chi_0 = np.array([chi_0, 1-chi_0])

chi_t = f_NN(Dt)
chi_t = pt.mean( chi_t, axis=1 ).cpu().detach().numpy()
chi_t = np.array([chi_t, 1-chi_t])

rate12, rate21  = exit_rates_from_chi((frame + 1) * 100, 100 * 0.002, chi_0, chi_t)



fig = plt.figure(figsize=(10*in2cm, 10*in2cm))

ax  = fig.add_subplot(1, 1, 1)
ax.set_xlabel(r'$\chi_0$')
ax.set_ylabel(r'$\chi_{\tau}}$')

ax.plot(chi_0[0,:], chi_t[0,:],   '.')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.2, hspace=0.5)
fig.savefig('chi_0-chi_t.png', format='png', dpi=300, bbox_inches='tight')

