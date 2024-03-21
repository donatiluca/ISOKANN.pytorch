# directories
inp_dir  = 'input/'
out_dir  = 'output/'



import mdtraj as md
import numpy as np
import sympy as sp
import torch as pt
from tqdm import tqdm
import scipy
import itertools
import matplotlib.pyplot as plt
import glob

np.random.seed(0)
pt.manual_seed(0)

font = {'size'   : 10}
plt.rc('font', **font)
in2cm = 1/2.54  # centimeters in inches

cuda = pt.device('cuda')
print("Cuda?")
print(pt.cuda.is_available())



def generate_pairs(N):
    #return np.c_[np.array(np.meshgrid(np.arange(N), np.arange(N))).T.reshape(-1,2)]
    t = np.arange(0,N,1)
    return np.array(list(set(itertools.combinations(t, 2))))

pairs =   generate_pairs(73)


# In[9]:


# Load initial states and plot relevant coordinate
Ndims   = len(pairs)
Npoints = 500
R0 = np.empty(Npoints, dtype = object)

for i in range(Npoints):
    
    x0     =  md.load(inp_dir + 'initial_states/x0_' + str(i) + '.pdb', 
                top = inp_dir + 'initial_states/x0_' + str(i) + '.pdb') 
    r0     =  md.compute_distances(x0, [[0,70]], periodic=False)
    R0[i]  =  r0

np.savetxt('R0.txt', R0)


# Load initial states

D0 = pt.empty((Npoints, Ndims), dtype = pt.float32, device=cuda)

for i in range(Npoints):
    print(i)
    x0     =  md.load(inp_dir + 'initial_states/x0_' + str(i) + '.pdb', 
                top = inp_dir + 'VGVAPG_nowat.prmtop')  
    d0       =  md.compute_distances(x0, pairs, periodic=False)
    D0[i,:]  =  pt.tensor(d0, dtype=pt.float32, device=cuda)

print(D0.shape)

pt.save(D0,'D0.pt')


# Load one final states
Nfinpoints  = 100
Ntimesteps = 10

Dt = pt.empty((Npoints, Nfinpoints, Ndims, Ntimesteps), dtype = pt.float32, device=cuda)

for i in range(Npoints):
    print(i)
    #list_files_i = glob.glob(out_dir + "final_states/xf_" + str(i) + "_r*.dcd")
    
    for j in range(Nfinpoints):
        xt         =  md.load(out_dir + "final_states/xf_" + str(i) + "_r" + str(j) + ".dcd", top = inp_dir + "VGVAPG_nowat.prmtop")

        #xt         =  md.load(list_files_i[j], 
        #                top = inp_dir + "VGVAPG_nowat.prmtop")
        for k in range(Ntimesteps):
            dt         =  md.compute_distances(xt[k], pairs, periodic=False)
            Dt[i,j,:,k]  =  pt.tensor(dt, dtype=pt.float32, device=cuda)

print(Dt.shape)

pt.save(Dt,'Dt.pt')
