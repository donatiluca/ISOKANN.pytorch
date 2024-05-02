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


print(" ")
cuda = pt.device('cuda')
print("Cuda?")
print(pt.cuda.is_available())
print(" ")
print("This will create tensors for ISOKANN using only backbone atoms")
print(" ")



def generate_pairs(N):
    #return np.c_[np.array(np.meshgrid(np.arange(N), np.arange(N))).T.reshape(-1,2)]
    t = np.arange(0,N,1)
    return np.array(list(set(itertools.combinations(t, 2))))


# directories
inp_dir  = '../input/'
out_dir  = '../output/'
print(">> Working directories:")
print("Input files:", inp_dir)
print("Output files:", out_dir)
print(" ")


# Starting points (number of files in input/initial_states)
traj     = md.load(out_dir + "trajectory_water.dcd", top = inp_dir + "pdbfile_water.pdb")   
Npoints  = traj.n_frames
print("Number of initial states:", Npoints)

_, _, files = next(os.walk(out_dir + 'final_states/'))
Nfinpoints = int(( len(files) - 1 ) / Npoints)
print("Number of final states states:", Nfinpoints)


pdb = PDBFile(inp_dir + 'pdbfile_no_water.pdb')
Natoms = pdb.topology.getNumAtoms()
print("Number of atoms (no water):", Natoms)


# Select backbone atoms and # Generate atoms pairs
bb    = traj.topology.select("backbone")
pairs   =   generate_pairs(len(bb))
Ndims   =   len(pairs)
print("Number of pairwise distances (only backbone atoms):", Ndims)


# Calculate relevant coordinate
print("I am generating the relevant coordinate...")
r        = md.compute_distances(traj,[[0,Natoms]]) # distance between first and last atom
np.savetxt('R0.txt', r)

print(" ")

# Load initial states
print("I am creating the tensor with the initial states...")
d0  =  md.compute_distances(traj.atom_slice(bb), pairs, periodic=False)
D0  =  pt.tensor(d0, dtype=pt.float32, device=cuda)

print('Shape of D0?')
print('Npoints, Ndims')
print(D0.shape)
print(" ")

pt.save(D0,'D0.pt')


# Load one trajectory to calculate number of frames
print("I am creating the tensor with the final states...")
xt         =  md.load(out_dir + "final_states/xf_0_r0.dcd", 
                              top = inp_dir + "pdbfile_no_water.pdb")
Ntimesteps = xt.n_frames

Dt = pt.empty((Npoints, Nfinpoints, Ndims, Ntimesteps), dtype = pt.float32, device=cuda)

for i in tqdm(range(Npoints)):
    
    for j in range(Nfinpoints):
        xt         =  md.load(out_dir + "final_states/xf_" + str(i) + "_r" + str(j) + ".dcd", 
                              top = inp_dir + "pdbfile_no_water.pdb")

        #xt         =  md.load(list_files_i[j], 
        #                top = inp_dir + "VGVAPG_nowat.prmtop")
        for k in range(Ntimesteps):
            dt         =  md.compute_distances(xt.atom_slice(bb)[k], pairs, periodic=False)
            Dt[i,j,:,k]  =  pt.tensor(dt, dtype=pt.float32, device=cuda)


print(" ")
print('Shape of Dt?')
print('Npoints, Nfinpoints, Ndims, Nframes')
print(Dt.shape)

pt.save(Dt,'Dt.pt')

