import mdtraj as md
import matplotlib.pyplot as plt
import numpy as np

# directories

inp_dir  =  'input/'
out_dir  =  'output/'

traj     = md.load(out_dir + "trajectory_water.dcd", top = inp_dir + "pdbfile_water.pdb")   

traj[0].save_gro(inp_dir + 'topology_water.gro')


import MDAnalysis as mda

traj = mda.Universe('topology.psf', out_dir + "trajectory_water.dcd")
