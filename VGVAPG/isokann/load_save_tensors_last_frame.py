import os
import mdtraj as md
import numpy as np
import sympy as sp
import torch as pt
from tqdm import tqdm
import scipy
import itertools
import matplotlib.pyplot as plt
import glob
from openmm import *
from openmm.app import *
from openmm.unit import *

font = {'size'   : 10}
plt.rc('font', **font)
in2cm = 1/2.54  # centimeters in inches

np.random.seed(0)
pt.manual_seed(0)


cuda = pt.device('cuda')
print("Cuda?")
print(pt.cuda.is_available())


def generate_pairs(N):
    #return np.c_[np.array(np.meshgrid(np.arange(N), np.arange(N))).T.reshape(-1,2)]
    t = np.arange(0,N,1)
    return np.array(list(set(itertools.combinations(t, 2))))


# directories
inp_dir  = '../input/'
out_dir  = '../output/'


# Starting points (number of files in input/initial_states)
traj     = md.load(out_dir + "trajectory_water.dcd", top = inp_dir + "pdbfile_water.pdb")   
Npoints  = traj.n_frames
print("Number of initial states:", Npoints)

# Final points
_, _, files = next(os.walk(out_dir + 'final_states/'))
Nfinpoints = int(( len(files) - 1 ) / Npoints)
print("Number of final states states:", Nfinpoints)


pdb = PDBFile(inp_dir + 'pdbfile_no_water.pdb')
Natoms = pdb.topology.getNumAtoms()
print("Number of atoms (no water):", Natoms)

pairs =   generate_pairs(Natoms)

# Load initial states and plot relevant coordinate
Ndims   = len(pairs)

R0 = np.zeros(Npoints)

for i in tqdm(range(Npoints)):

    r        = md.compute_distances(traj[i],[[0,Natoms]]) # distance between first and last atom
    R0[i]  =  r

np.savetxt('R0.txt', R0)

# Load initial states
D0 = pt.empty((Npoints, Ndims), dtype = pt.float32, device=cuda)

for i in range(Npoints):
    print(i)

    d0       =  md.compute_distances(traj[i], pairs, periodic=False)
    D0[i,:]  =  pt.tensor(d0, dtype=pt.float32, device=cuda)

print(D0.shape)

pt.save(D0,'D0.pt')


# Load one trajectory to calculate number of frames
Ntimesteps = 1

Dt = pt.empty((Npoints, Nfinpoints, Ndims, Ntimesteps), dtype = pt.float32, device=cuda)

for i in range(Npoints):
    print(i)
    
    for j in range(Nfinpoints):
        xt         =  md.load(out_dir + "final_states/xf_" + str(i) + "_r" + str(j) + ".dcd", 
                              top = inp_dir + "pdbfile_no_water.pdb")

        #xt         =  md.load(list_files_i[j], 
        #                top = inp_dir + "VGVAPG_nowat.prmtop")
        
        # LOAD ONLY THE LAST FRAME (WE SAVED 10 FRAMES)
        k            =  9 
        dt           =  md.compute_distances(xt[k], pairs, periodic=False)
        Dt[i,j,:,0]  =  pt.tensor(dt, dtype=pt.float32, device=cuda)

print(Dt.shape)

pt.save(Dt,'Dt.pt')
