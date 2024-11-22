from openmm import *
from openmm.app import *
from openmm.unit import *
from sys import stdout
import mdtraj.reporters
import numpy as np
from time import gmtime, strftime

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

# Integrator parameters
dt       = 0.002      # ps

# Length initial trajectories (integrator timesteps)
Nsteps   = 5000                                          #500000000

# Number of frames to save (only molecule)
Nframes    = 500 

# Number of frames to save (molecule + water)
# These will be the number of initial points for the second set of simulations
NframesW  = 5                                             #2000

# Rates at which frames are saved
Nout     = int(Nsteps / Nframes) 
NoutW    = int(Nsteps / NframesW) 

print(">> Integrator parameters:")
print("Integrator timestep:", str(dt), "ps")
print("Number of timesteps:", str(Nsteps))
print('Real time:', str(Nsteps * dt / 1000), "ns")
print("Saved frames (only molecule):", str(Nframes))
print("Saved frames (molecule+water):", str(NframesW))
print(" ")

# System parameters
kB    = 0.008314  # kJ mol-1 K-1 
T     = 300       # K
gamma = 1         # ps-1

print(">> System parameters:")
print("Temperature:", str(T), "K")
print("Friction:", str(gamma), "ps-1\n")


# Generate log-file
log = open(out_dir                                 + "log.txt", 'w')
log.write('Timestep: '                             + str(dt)     + " ps\n")
log.write("nsteps: "                               + str(Nsteps) + "\n" )
log.write("Saved frames (only molecule): "         + str(Nframes)   + "\n")
log.write("Saved frames (molecule + water): "      + str(NframesW)   + "\n")
log.write("nstxout: "                              + str(Nout)   + "\n")
log.write("Temperature: "                          + str(T)      + " K\n")
log.write("Collision rate: "                       + str(gamma)  + " ps-1\n")
log.write("Boltzmann const.: "                     + str(kB)     + " kJ mol-1 K-1 \n")
log.write("Simulation start: "                     + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n" )
log.close();

# Load molecule files
pdb = PDBFile(inp_dir + 'pdbfile_no_water.pdb')
Natoms = pdb.topology.getNumAtoms()

# Load forcefield
forcefield = ForceField("amber14/protein.ff14SB.xml", "amber14/tip3pfb.xml")

# Solvation
print("Solvation ... \n")

modeller = Modeller(pdb.topology, pdb.positions)

# cubic water box with a minimum distance of 1 nm to the box boarders
modeller.addSolvent(forcefield, padding=1*nanometer, neutralize=True)
pdb.positions = modeller.getPositions()
pdb.topology  = modeller.getTopology()

with open(inp_dir + "pdbfile_water.pdb", "w") as file_:
    pdb.writeFile(
        pdb.topology, pdb.positions,
        file=file_
    )

# Load pdbfile with water
pdb = PDBFile(inp_dir + 'pdbfile_water.pdb')

print(">> Molecule parameters:")
print("Number of atoms (no water):", Natoms)
print("Number of water molecules:", pdb.topology.getNumAtoms() - Natoms)
print("Total number of molecules:", pdb.topology.getNumAtoms())
print(" ")

# create system
"""
# Implicit solvent
system = forcefield.createSystem( pdb.topology, 
                                  nonbondedMethod=PME, 
                                  nonbondedCutoff=1.0*nanometer, 
                                  constraints=HBonds )
"""

# Explicit solvent
system = forcefield.createSystem( pdb.topology, 
                                  nonbondedMethod=CutoffNonPeriodic,
                                  nonbondedCutoff=1 * nanometer,
                                  constraints=None    )


#system = prmtop.createSystem(implicitSolvent=app.OBC2, nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=1.0, constraints=None, rigidWater=True)


integrator = LangevinIntegrator(T*kelvin, gamma/picosecond, dt*picoseconds)

##########################################################################################################
###  S T A R T   S I M U L A T I O N  ####################################################################
##########################################################################################################

# set-up simulation
platform = Platform.getPlatformByName('CUDA')
simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

# minimization
print('\n\n*** Minimizing ...')
simulation.minimizeEnergy()
print('*** Minimization completed ***') 

# equilibration
simulation.context.setVelocitiesToTemperature(T)
print('\n\n*** Equilibrating...')

simulation.step(10000)
print('*** Equilibration completed ***')




##########################################################################################################
###  S A V E S  ##########################################################################################
##########################################################################################################

# print on screen
simulation.reporters.append(StateDataReporter(stdout, 10000, speed = True, step=True, potentialEnergy=True, temperature=True))

# save trajectory
simulation.reporters.append(mdtraj.reporters.DCDReporter(out_dir + "trajectory_water.dcd", NoutW))

simulation.reporters.append(mdtraj.reporters.DCDReporter(out_dir + "trajectory.dcd", Nout, atomSubset=range(Natoms)))


# repeat procedure for nsteps
simulation.step(Nsteps)

# add total calculation time to LOG-file
log = open(out_dir + "log.txt", 'a')
log.write("Simulation end: " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) )
log.close();

# end
print('\n\n****** SIMULATION COMPLETED *****************************\n\n')