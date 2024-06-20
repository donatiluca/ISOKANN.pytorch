from openmm import *
from openmm.app import *
from openmm.unit import *
from sys import stdout
import mdtraj.reporters
import numpy as np
from time import gmtime, strftime
import mdtraj as md

print(" ")
print(">> Simulation is starting, check the parameters ... ")
print(" ")

# directories
inp_dir  =  'input/'
out_dir  =  'output/'

print(">> Working directories:")
print("Input files:", inp_dir)
print("Output files:", out_dir)
print(" ")

# Starting points (number of files in input/initial_states)
traj_water       = md.load_dcd(out_dir + 'trajectory_water.dcd', top=inp_dir + 'pdbfile_water.pdb')
pdbfile_no_water = PDBFile(inp_dir + 'pdbfile_no_water.pdb') # this file is used to count atoms of the molecule
Natoms           = pdbfile_no_water.topology.getNumAtoms()
Npoints          = traj_water.n_frames
print("Number of initial states:", Npoints)

# Number of replicas (the number of files in output/final_states will be Npoints x NfinPoints)
NfinPoints = 10
print("Number of final states:", NfinPoints)
print(" ")

# Integrator parameters
dt      = 0.001      # ps
Nsteps  = 2000

# Number of frames to save (only molecule)
Nframes    = 10 

# Rate at which frames are saved
Nout     = int(Nsteps / Nframes) 

print(">> Integrator parameters:")
print("Integrator timestep:", str(dt), "ps")
print("Number of timesteps:", str(Nsteps))
print('Real time:', str(Nsteps * dt / 1000), "ns")
print("Saved frames (only molecule):", str(Nframes))
print(" ")


# System parameters
kB    = 0.008314  # kJ mol-1 K-1 
T     = 300       # K
gamma = 1         # ps-1

print(">> System parameters:")
print("Temperature:", str(T), "K")
print("Friction:", str(gamma), "ps-1\n")

# LOG-file
log = open(out_dir                                 + "final_states/log.txt", 'w')
log.write("Number of initial states: "             + str(Npoints) + "\n" )
log.write("Number of final states: "               + str(NfinPoints) + "\n" )
log.write('Timestep: '                             + str(dt)     + " ps\n")
log.write("nsteps: "                               + str(Nsteps) + "\n" )
log.write("Saved frames (only molecule): "         + str(Nframes)   + "\n")
log.write("Temperature: "                          + str(T)      + " K\n")
log.write("Collision rate: "                       + str(gamma)  + " ps-1\n")
log.write("Boltzmann const.: "                     + str(kB)     + " kJ mol-1 K-1 \n")
log.write("Simulation start: "                     + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n" )
log.close();

forcefield = ForceField("amber14/protein.ff14SB.xml", "amber14/tip3pfb.xml")


pdb = PDBFile(inp_dir + 'pdbfile_water.pdb')
system = forcefield.createSystem(pdb.topology, 
                                 nonbondedMethod=PME, 
                                 nonbondedCutoff=1.0*nanometer, 
                                 constraints=HBonds)


print(">> Molecule parameters:")
print("Number of atoms (no water):", Natoms)
print("Number of water molecules:", pdb.topology.getNumAtoms() - Natoms)
print("Total number of molecules:", pdb.topology.getNumAtoms())
print(" ")

for i in range(Npoints):
    #i=36
    print("Initial state:", i)
    
    # Get the positions of the atoms from the current frame
    positions = traj_water.xyz[i]
    


    # Num replicas per initial state
    for r in range(NfinPoints):
        print("Replica:", r)    
        
        # set-up simulation
        integrator = LangevinMiddleIntegrator(T*kelvin, gamma/picosecond, dt*picoseconds)
        platform = Platform.getPlatformByName('CUDA')    


        #properties = {'Precision': 'double', 'DisablePmeStream':'true'}
        simulation = Simulation(pdb.topology, system, integrator, platform)
        simulation.context.setPositions(pdb.positions)
        #simulation.minimizeEnergy(maxIterations=100)

        simulation.context.setVelocitiesToTemperature(T)
        
        # print on screen
        #simulation.reporters.append(StateDataReporter(stdout, 10, speed = True, step=True, potentialEnergy=True, temperature=True))
        
        # save trajectory
        #simulation.reporters.append(DCDReporter(out_dir + 'final_states/xf_' + str(i) + '_r' + str(r) + '.dcd', Nout))
        simulation.reporters.append(mdtraj.reporters.DCDReporter(out_dir + 'final_states/xf_' + str(i) + '_r' + str(r) + '.dcd', Nout, atomSubset=range(Natoms)))
        # repeat procedure for nsteps
        simulation.step(Nsteps)

# add total calculation time to LOG-file
log = open(out_dir + 'final_states/log.txt', 'a')
log.write("Simulation end: " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) )
log.close();

# end
print('\n\n****** SIMULATION COMPLETED *****************************\n\n')
