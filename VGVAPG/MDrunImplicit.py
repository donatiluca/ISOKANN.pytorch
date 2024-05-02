from openmm import *
from openmm.app import *
from openmm.unit import *
from sys import stdout
import mdtraj.reporters
import numpy as np
from time import gmtime, strftime

# directories
inp_dir  =  'input/'
out_dir  =  'output/'

# Integrator parameters
dt      = 0.002      # ps
Nsteps  = 50000000 #2500000
#Npoints = 500
Nout    = 500#int( Nsteps / Npoints )
print("Nout = ", Nout)

print('*** Number of nanoseconds ***')
print(Nsteps * dt / 1000)

# System parameters
kB    = 0.008314  # kJ mol-1 K-1 
T     = 300       # K
gamma = 1         # ps-1


# LOG-file
log = open(out_dir             + "log.txt", 'w')

log.write('Timestep: '         + str(dt)     + " ps\n")
log.write("nsteps: "           + str(Nsteps) + "\n" )
log.write("nstxout: "          + str(Nout)   + "\n")
log.write("Temperature: "      + str(T)      + " K\n")
log.write("Collision rate: "   + str(gamma)  + " ps-1\n")
log.write("Boltzmann const.: " + str(kB)     + " kJ mol-1 K-1 \n")
log.write("Simulation start: " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n" )
log.close();

pdb        = PDBFile(inp_dir + 'pdbfile.pdb')
forcefield = ForceField('amber99sbildn.xml', 'implicit/obc2.xml')


# create system
system = forcefield.createSystem(   pdb.topology, 
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
#simulation.reporters.append(DCDReporter(out_dir + "trajectory.dcd", Nout, enforcePeriodicBox=True))

simulation.reporters.append(mdtraj.reporters.DCDReporter(out_dir + "trajectory.dcd", Nout))


# repeat procedure for nsteps
simulation.step(Nsteps)

# add total calculation time to LOG-file
log = open(out_dir + "log.txt", 'a')
log.write("Simulation end: " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) )
log.close();

# end
print('\n\n****** SIMULATION COMPLETED *****************************\n\n')
