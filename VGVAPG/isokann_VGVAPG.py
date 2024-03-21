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
import glob
from functionsNN import NeuralNetwork, trainNN, exit_rates_from_chi

np.random.seed(0)
pt.manual_seed(0)


font = {'size'   : 10}
plt.rc('font', **font)
in2cm = 1/2.54  # centimeters in inches


def scale_and_shift(y):

    minarr = np.min(y)
    maxarr = np.max(y)
    hat_y  = (y - minarr) / (maxarr - minarr)

    return hat_y


if pt.cuda.is_available():
    dev = pt.device('cuda')
else:
    dev = pt.device('cpu')

print("Cuda?")
print(pt.cuda.is_available())


# Load initial states
D0 = pt.load(main_dir + 'D0.pt', map_location=dev)
print('Shape of D0?')
print('Npoints, Ndims')
print(D0.shape)
print(" ")

Npoints = D0.shape[0]
Ndims   = D0.shape[1]

# Load relevant coordinate
R0 = np.loadtxt(main_dir + 'R0.txt')

# Load final states
DT = pt.load(main_dir + 'Dt.pt', map_location=dev)
print('Shape of Dt?')
print('Npoints, Nfinpoints, Ndims, Nframes')
print(DT.shape)
Nfinpoints  = DT.shape[1]
Nframes     = DT.shape[3]

# Select a fram between 0 and 9
frame = 9

# Select only final states between 0 and 10
Dt = pt.clone(DT[:,0:10,:,frame])
print(" ")
print("frame=", frame)
# Define the NN

nodes = np.array([Ndims, 204, 102, 51, 1])
#nodes = np.array([Ndims, 1200, 1])
np.savetxt('nodes.txt', nodes)

f_NN = NeuralNetwork( Nodes = nodes, enforce_positive = 0 ).to(dev)

Niters = 50

LOSS = np.empty(0, dtype = object)

for i in tqdm(range(Niters)):
    pt_chi =  f_NN(Dt)
    pt_y   =  pt.mean(pt_chi, axis=1)
    y      =  scale_and_shift(pt_y.cpu().detach().numpy())
    pt_y   =  pt.tensor(y, dtype=pt.float32, device=dev) # , requires_grad = False
    loss1 = trainNN(net = f_NN, lr = 1e-4, wd = 1e-10, Nepochs = 10, batch_size=100, X=D0, Y=pt_y)
    LOSS  = np.append(LOSS, loss1)


chi       =  f_NN(D0).cpu().detach().numpy()

fig = plt.figure(figsize=(24*in2cm, 10*in2cm))

ax = fig.add_subplot(1, 2, 1)
ax.plot(R0, chi,   'sb')
ax.plot(R0, 1-chi,   'sr')


ax.set_xlabel(r'$r_0$ / nm')
ax.set_ylabel(r'$\chi_i$')
ax.set_title('Membership functions')
ax.set_ylim(-0.1,1.1);

ax = fig.add_subplot(1, 2, 2)
ax.plot(LOSS)
ax.set_yscale('log')
ax.set_xlabel('Epoch')

ax.set_title('Loss function')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.2, hspace=0.5)
fig.savefig('chi.png', format='png', dpi=300, bbox_inches='tight')



pt.save(f_NN.state_dict(),'f_NN.pt')
np.savetxt('LOSS.txt', LOSS)
np.savetxt('chi.txt', chi)




print(" ")
print(" E N D ")
print(" ")
