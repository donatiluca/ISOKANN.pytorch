from openmm import *
from openmm.app import *
from openmm.unit import *
from sys import stdout
import mdtraj.reporters
import numpy as np
from time import gmtime, strftime
import multiprocessing
from joblib import Parallel, delayed
import mdtraj 
import os

Njobs = int(multiprocessing.cpu_count())
print('Njobs = ', str(Njobs))


# directories
inp_dir  =  'input/'
out_dir  =  'output/'

# Starting points (number of files in input/initial_states)
_, _, files = next(os.walk(inp_dir + 'initial_states/'))
Npoints = len(files)
print("Number of initial states:", Npoints)



# Number of replicas (the number of files in output/final_states will be Npoints x NfinPoints)
NfinPoints = 10
print("Number of final states:", NfinPoints)






# Integrator parameters
dt      = 0.002      # ps
Nsteps  = 1000
Nout    = 100


print('*** Number of nanoseconds ***')
print(Nsteps * dt / 1000)

# System parameters
kB    = 0.008314  # kJ mol-1 K-1 
T     = 300       # K
gamma = 1         # ps-1


# LOG-file
log = open(out_dir             + "final_states/log.txt", 'w')

log.write('Timestep: '         + str(dt)     + " ps\n")
log.write("nsteps: "           + str(Nsteps) + "\n" )
log.write("nstxout: "          + str(Nout)   + "\n")
log.write("Temperature: "      + str(T)      + " K\n")
log.write("Collision rate: "   + str(gamma)  + " ps-1\n")
log.write("Boltzmann const.: " + str(kB)     + " kJ mol-1 K-1 \n")
log.write("Simulation start: " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n" )
log.close();

forcefield = ForceField('amber99sbildn.xml', 'implicit/obc2.xml')
platform = Platform.getPlatformByName('Reference')



def simulation(i, r, Nsteps, T, gamma, platform, forcefield):
    
    if r==0:
        print(i)

    
    pdb    = PDBFile(inp_dir + 'initial_states/x0_' + str(i) + '.pdb')

    system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=CutoffNonPeriodic,
            nonbondedCutoff=1 * nanometer,
            constraints=None
            )
    
    integrator = LangevinIntegrator(T*kelvin, gamma/picosecond, dt*picoseconds)

    simulation = Simulation(pdb.topology, system, integrator, platform)

    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(T)
    

    # save trajectory
    simulation.reporters.append(mdtraj.reporters.DCDReporter(out_dir + 'final_states/xf_' + str(i) + '_r' + str(r) + '.dcd', Nout))

    simulation.step(Nsteps)
    #reporter.close()



Parallel(n_jobs=Njobs, prefer="threads")(delayed(simulation)(i, r, Nsteps, T, gamma, platform, forcefield) for i in range(Npoints) for r in range(NfinPoints))




# add total calculation time to LOG-file
log = open(out_dir + 'final_states/log.txt', 'a')
log.write("Simulation end: " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) )
log.close();

# end
print('\n\n****** SIMULATION COMPLETED *****************************\n\n')