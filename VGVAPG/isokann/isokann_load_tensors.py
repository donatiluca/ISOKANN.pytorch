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
import glob
from functionsNN import NeuralNetwork, trainNN, exit_rates_from_chi
import cmocean 
import matplotlib.cm as cm

#np.random.seed(0)
#pt.manual_seed(0)


font = {'size'   : 10}
plt.rc('font', **font)
in2cm = 1/2.54  # centimeters in inches


def scale_and_shift(y):

    minarr = np.min(y)
    maxarr = np.max(y)
    hat_y  = (y - minarr) / (maxarr - minarr)

    return hat_y

def cluster_by_isa(Evs, NoOfClus = 2):
    
    if NoOfClus < 2:
        Chi      =  np.ones(Evs.shape[0])
        indic    = 0
        RotMat   = 1 / Evs[0,0]
        cF       = np.ones(Evs.shape[0])
    elif NoOfClus >= Evs.shape[0] and Evs.shape[1] == Evs.shape[0]:
        Chi      = np.eye(Evs.shape[0])
        indic    = 0
        RotMat   = np.linalg.inv(Evs)
        cF       = np.arange(Evs.shape[0])
    else:
        NoOfClus = np.round(NoOfClus)
        
        if NoOfClus > Evs.shape[1]:
            NoOfClus = Evs.shape[1]
        
        C        = Evs[:, 0:NoOfClus]
        
        OrthoSys = np.copy(C)
        maxdist  = 0.0
    
        ind = np.zeros(NoOfClus, dtype=int)
        for i in range(Evs.shape[0]): 
            if np.linalg.norm(C[i,:]) > maxdist:
                maxdist = np.linalg.norm(C[i,:])
                ind[0]  = i
        
        for i in range(Evs.shape[0]): 
            OrthoSys[i,:] = OrthoSys[i,:] - C[ind[0],:]
        
        for k in range(1, NoOfClus):
            maxdist = 0.0
            temp    = np.copy(OrthoSys[ind[k-1],:])

            for i in range(Evs.shape[0]): 
                
                OrthoSys[i,:] = OrthoSys[i,:] - np.matmul(temp , OrthoSys[i,:].T) * temp 
                distt         = np.linalg.norm(OrthoSys[i,:])
                #print(distt)
                if distt > maxdist:
                    maxdist = np.copy(distt)
                    ind[k]  = i
                    
            OrthoSys = OrthoSys / np.linalg.norm(OrthoSys[ind[k],:])
        
        
        RotMat = np.linalg.inv(C[ind,:])
        Chi    = np.matmul( C, RotMat )
        indic  = np.min(Chi)
        #[minVal cF] = np.max(transpose(Chi))
    
        
    return Chi, RotMat


cuda = pt.device('cuda')
print("Cuda?")
print(pt.cuda.is_available())
print(" ")


# Load initial states
D0 = pt.load('D0.pt', map_location=cuda)
print('Shape of D0?')
print('Npoints, Ndims')
print(D0.shape)
print(" ")

Npoints = D0.shape[0]
Ndims   = D0.shape[1]

# Load relevant coordinates
R0 = np.loadtxt('R0.txt')

# Load final states
DT = pt.load('Dt.pt', map_location=cuda)
print('Shape of Dt?')
print('Npoints, Nfinpoints, Ndims, Nframes')
print(DT.shape)
Nfinpoints  = DT.shape[1]
Nframes     = DT.shape[3]
frame = 0

Dt = pt.clone(DT[:,:,:,frame])
print(" ")
print("frame=", frame)


# Define the NN
NNnodes = np.array([Ndims, 1200, 1])
np.savetxt('NNnodes.txt', NNnodes)
f_NN = NeuralNetwork( Nodes = NNnodes, enforce_positive = 0 ).to(cuda)



Niters = 2000

LOSS = np.empty(0, dtype = object)

for i in tqdm(range(Niters)):
    pt_chi =  f_NN(Dt)
    pt_y   =  pt.mean(pt_chi, axis=1)
    y      =  scale_and_shift(pt_y.cpu().detach().numpy())
    pt_y   =  pt.tensor(y, dtype=pt.float32, device=cuda) # , requires_grad = False
    loss1 = trainNN(net = f_NN, lr = 5e-3, wd = 1e-10, Nepochs = 1, batch_size=200, X=D0, Y=pt_y)
    LOSS  = np.append(LOSS, loss1)


print(" ")


chi       =  f_NN(D0).cpu().detach().numpy()



fig = plt.figure(figsize=(24*in2cm, 10*in2cm))

ax = fig.add_subplot(1, 2, 1)
ax.scatter(np.arange(Npoints), R0, c=chi,  cmap = cm.RdYlBu_r , s = 50 )

ax.set_xlabel(r'$r$ / nm')
ax.set_ylabel(r'$\chi$')
ax.set_title('Membership function')


ax = fig.add_subplot(1, 2, 2)
ax.plot(LOSS)
ax.set_yscale('log')
ax.set_xlabel('Epoch')

ax.set_title('Loss function')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.2, hspace=0.5)
fig.savefig('results/chi_' + str(frame) + '.png', format='png', dpi=300, bbox_inches='tight')



pt.save(f_NN.state_dict(),'results/f_NN_' + str(frame) + '.pt')
np.savetxt('results/LOSS_' + str(frame) + '.txt', LOSS)
np.savetxt('results/chi_' + str(frame) + '.txt', chi)



del f_NN

print(" ")
print(" E N D ")
print(" ")
