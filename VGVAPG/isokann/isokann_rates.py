print(" ")
print(" S T A R T ")
print(" ")

# directories

inp_dir  =  '../input/'
out_dir  =  '../output/'

import mdtraj as md
import numpy as np
import sympy as sp
import torch as pt
from tqdm import tqdm
import scipy
import itertools
import matplotlib.pyplot as plt
from functionsNN import NeuralNetwork, trainNN, exit_rates_from_chi

cuda = pt.device('cuda')
print("Cuda?")
print(pt.cuda.is_available())
print(" ")


font = {'size'   : 10}
plt.rc('font', **font)
in2cm = 1/2.54  # centimeters in inches



# Load initial states
D0 = pt.load('D0.pt', map_location=cuda)
Npoints = D0.shape[0]
Ndims   = D0.shape[1]
print('Shape of D0?')
print('Npoints, Ndims')
print(D0.shape)
print(" ")

# Load final states
DT = pt.load('Dt.pt', map_location=cuda)
Nfinpoints  = DT.shape[1]
print('Shape of Dt?')
print('Npoints, Nfinpoints, Ndims')
print(DT.shape)

rates = np.zeros((2,10))


fig = plt.figure(figsize=(10*in2cm, 10*in2cm))
ax  = fig.add_subplot(1, 1, 1)
ax.set_xlabel(r'$\chi_0$')
ax.set_ylabel(r'$\chi_{\tau}$')

frame = 0
    
print(" ")
print("frame=", frame)

NNnodes = np.loadtxt('NNnodes.txt')
print(NNnodes)


Dt = pt.clone(DT[:,:,:,frame])

f_NN = NeuralNetwork( Nodes = NNnodes.astype(int), enforce_positive = 0 ).to(cuda)
f_NN.load_state_dict(pt.load('results/f_NN_' + str(frame) + '.pt'))

chi_0 = f_NN(D0).cpu().detach().numpy()


chi_0 = np.array([chi_0, 1-chi_0])

chi_t = f_NN(Dt)
chi_t = pt.mean( chi_t, axis=1 ).cpu().detach().numpy()
chi_t = np.array([chi_t, 1-chi_t])

rate12, rate21  = exit_rates_from_chi((frame + 1) * 100, 100 * 0.002, chi_0, chi_t)

rates[0,frame] = rate12
rates[1,frame] = rate21

ax.plot(chi_0[0,:], chi_t[0,:],   '.', label = r'$\tau=$'+str(np.round((frame+1)*0.2,1))+'ps')

del f_NN                        

ax.legend()
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.2, hspace=0.5)
fig.savefig('results/chi_0-chi_t.png', format='png', dpi=300, bbox_inches='tight')



tmprates= np.copy(rates)
rates[0,1] = tmprates[1,1]
rates[1,1] = tmprates[0,1]
time = np.linspace(1,10,10) * 0.002 * 100

fig = plt.figure(figsize=(10*in2cm, 10*in2cm))
ax  = fig.add_subplot(1, 1, 1)
ax.plot(time, rates[0,:], 'bs--', label=r'$k_{12}$')
ax.plot(time, rates[1,:], 'rs--', label=r'$k_{21}$')

ax.set_xlabel(r'$\tau$ / ps')
ax.set_ylabel(r'$k$ / ps$^{-1}$')
ax.legend()
ax.set_ylim(0.000001, 0.0001)
ax.set_yscale('log')



plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.2, hspace=0.5)
fig.savefig('results/rates.png', format='png', dpi=300, bbox_inches='tight')