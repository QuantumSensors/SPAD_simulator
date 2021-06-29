# 2D Monte Carlo simulator for modeling the operation of single-photon avalanche detectors.
# Additional documentation can be found at https://github.com/QuantumSensors 
# Key references:
# [1] S. Yanikgonul, V. Leong, J. R. Ong, C. E. Png, and L. Krivitsky, "2D Monte Carlo simulation of a silicon waveguide-based single-photon avalanche diode for visible wavelengths," Opt. Express 26, 15232-15246 (2018). (https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-26-12-15232&id=390108)
# [2] S. Yanikgonul, V. Leong, J. R. Ong, C. E. Png and L. Krivitsky, "Simulation of Silicon Waveguide Single-Photon Avalanche Detectors for Integrated Quantum Photonics," IEEE Journal of Selected Topics in Quantum Electronics 26, 1-8 (2020). (https://ieeexplore.ieee.org/document/8820038)

# This code is split into 9 sections:
# 1) Imports - Import various libraries needed
# 2) Debugging - Variables and functions used for debugging
# 3) Reading input files - Parses and reads the 3 data files
# 4) Variables - Physical constants, info from the data files, definition of charge carrier class
# 5) Functions - Misc functions and preliminary routines 
# 6) Random Walk - Random walk routine if injection happens in neutral regions
# 7) Random Path Length - RPL routine for charge carriers in depletion region
# 8) Main simulation routine - Generates injection point, calls randwalk routine (if required), calls RPL over all charges in depletion region for max timesteps
# 9) Plotting - Matplotlib settings for plotting the device edges, depletion edges, injection point and random walk final point. Only runs if Test Suite is set to True. 

# Original authors: Salih Yanikgonul, Victor Leong, Junrong Ong
# Additional contributions: Gundlapalli Prithvi, Dillon Lim
# Agency for Science, Technology and Research (A*STAR), Singpaore
# 
# Correspondence should be directed to victor_leong@imre.a-star.edu.sg
#
# V1.0 released June 2021
#############

#############
## Imports ##
#############

import numpy as np
import pandas as pd
import random
import math
from itertools import count
from scipy import constants
import matplotlib
matplotlib.use('Agg') 
import argparse, sys
import time 
start_time = time.time()
# -----------------------------------------------------------------------------------------------------------------------------

###############
## Debugging ## 
###############

test_suite = False  # Set to True if want to debug 
if test_suite:
    import matplotlib.pyplot as plt
test_injection_coord = [-0.5e-5, -0.8e-5]
# -----------------------------------------------------------------------------------------------------------------------------

#########################
## Reading Input Files ##
#########################

parser = argparse.ArgumentParser()
parser.add_argument('-data_file')  # DC electrical characteristics
parser.add_argument('-pgen_file')  # Table of cumulative probabilities used to determine injection point 
parser.add_argument('-parameter_file')  # Other device and simulation parameters
if not test_suite:
    parser.add_argument('-run_number',type=str)  # Requires a run number if not debugging
args = parser.parse_args()

if not test_suite:
    print("#--------------")
    print("#--- Run no. "+args.run_number+" , time: "+time.strftime("%Y-%m-%d %H:%M"))
    print("#--------------")
    print("")
else:  # If you are debugging, manually key in file names
    args.data_file = 'data.out'
    args.pgen_file = 'pgen.dat'
    args.parameter_file = 'parameters.dat' 

sim_datafile = open(args.data_file,'r')
pgen_datafile = open(args.pgen_file,'r')
parameter_file = open(args.parameter_file, 'r')

pgen = pd.read_csv(pgen_datafile, delimiter=' ', header=None, comment='#').values
sim_data = pd.read_csv(sim_datafile, delimiter=' ', header=None, comment='#').values
parameters = pd.read_csv(parameter_file, delimiter=' ', header=None, comment='#').values 
# -----------------------------------------------------------------------------------------------------------------------------

###############
## Variables ##
###############

# --- Constants ---
e_charge = -1 * constants.e  # e is positive, -1 to indicate sign
e_ionization_threshold = 1.103 * constants.e  # Converts eV to J | Electron ionization threshold energy
h_ionization_threshold = 1.269 * constants.e  # Converts eV to J | Hole ionization threshold energy
cm_to_nm = 1e7

# --- Extracting info from data file ---
x = sim_data[:,0].astype(float)  # cm | coordinates
y = sim_data[:,1].astype(float)  # cm | coordinates
weight_Ex = sim_data[:,2].astype(float)  # V/cm | weighting field
weight_Ey = sim_data[:,3].astype(float)  # V/cm | weighting field
Ex = sim_data[:,4].astype(float)  # V/cm | electric field
Ey = sim_data[:,5].astype(float)  # V/cm | electric field
ion_e = sim_data[:,6].astype(float)  # - | electron ionization coefficient
ion_h = sim_data[:,7].astype(float)  # - | hole ionization coefficient
mob_e = sim_data[:,8].astype(float)  # cm^2/Vs | electron mobility
mob_h = sim_data[:,9].astype(float)  # cm^2/Vs | hole mobility
driftv_e = sim_data[:,10].astype(float)  # cm/s | electron drift velocity
driftv_h = sim_data[:,11].astype(float)  # cm/s | hole drift velocity

# --- Random Walk Parameters ---
temp = parameters[0][1]  # Temperature in K, used in diffusion calculation
randwalk_maxtime = parameters[0][3]   # s, How long we are willing to run randwalk for
randwalk_minsteps = parameters[0][4]  # -, Used to calculate randwalk step size

# --- RPL Parameters ---
dep_edge_efield_mag = parameters[0][0]  # V/cm, Threshold E-field to determine depletion edge boundary
avalanche_threshold = parameters[0][2]  # A, Threshold current to determine successful avalanche
RPL_timestep = parameters[0][5]  # s, Time interval for each step in RPL

# Print RPL timestep and avalanche threshold 
print("")
print(f"RPL timestep is {RPL_timestep*1e15} fs")
print(f"Avalanche threshold is {avalanche_threshold*1e6} uA")

# --- Device Geometry ---
x_core_left = parameters[0][6]  # cm
x_core_right = parameters[0][7]  # cm
wg_rib_y_coord = parameters[0][8]  # cm
y_slice = parameters[0][9]  # cm | nominal y-coordinate at the middle of the device rib. 
jnc_code = parameters[0][10]  # "OV" if outside, "IV" if inside vertical junction 

# --- Define simulation boundaries ---
x_uni, x_uni_idx = np.unique(x, return_index=True)  # Sorted array of unique elements in x and their indices
y_uni = np.unique(y)  # Sorted array of unique elements in y
x_device_right = x_uni[-1]  # cm
x_device_left = x_uni[0]  # cm
y_core_top = y_uni[-1]  # cm
y_core_bottom = y_uni[0]  # cm

# Print simulation region and device edges. 
# Simulation region covers all the data from the data file.
# Device edges are defined as the edges of rectangular core region.
print("")
print(f"Simulation region between x=[{x_device_left*1e7:.3f},{x_device_right*1e7:.3f}] nm and y=[{y_core_bottom*1e7:.3f},{y_core_top*1e7:.3f}] nm")
print(f"Device edges are x=[{x_core_left*1e7:.3f},{x_core_right*1e7:.3f}] nm and y=[{y_core_bottom*1e7:.3f},{y_core_top*1e7:.3f}] nm")

# --- Close data files ---
sim_datafile.close()
pgen_datafile.close()
parameter_file.close()
# -----------------------------------------------------------------------------------------------------------------------------

######################
## Helper Functions ##
######################

def lookup(pos):
    '''
    Maps the given position to the nearest [x,y] coordinate provided in the data file
    to perform a lookup of data at that coordinate. 
    
    Arguments
    ---------
    pos : array
        The current position in the form [x,y] in cm. 
    
    Returns
    -------
    field_idx : int
        The index of the nearest [x,y] coordinate in the data file. 
    '''
    idx = (np.abs(x_uni-pos[0])).argmin() 
    idy = (np.abs(y_uni-pos[1])).argmin() 
    field_idx = x_uni_idx[idx] + idy
    if field_idx<0 or field_idx>len(x):  # Bounds checking
        print('WARNING! Field array index out of bounds.')
        print(f"Pos: {pos} nm, Final index: {field_idx}, idx: {idx}, idy: {idy}, Max index: {len(x)}")
    return field_idx

def get_injection_point(pgen_array):
    '''
    Randomly generates the coordinates of the initial charge carrier injection point. 
    Called at the start of main simulation routine. 
    
    Arguments
    ---------
    pgen_array : array
        List that contains all the data from the pgen file. 
    
    Returns
    -------
    Injection point : array
        Coordinate in the form [x,y] in cm 
    '''
    r = random.uniform(0,1)   
    idx = np.where(pgen_array[:,2]>r)[0][0]       # Obtains first index where probability > r
    return [pgen[idx][0]*1e-4,pgen[idx][1]*1e-4]  # Convert units from um to cm

def idx_of_xcoord_at_max_field(y):
    '''
    Determines index of the x-coordinate with maximum E-field at a given y-coordinate.
    Primarily used to estimate the x-coordinate of the junction. 
    
    Arguments
    ---------
    y : float in cm
    
    Returns
    -------
    idx_of_xcoord_at_max_E_field : int
        Index of the x-coordinate that has the maximum E-field strength at that y value.
    '''
    max_E_field = 0
    idx_of_xcoord_at_max_E_field = 0
    for x in x_uni:
        idx = lookup([x,y])
        e_field_at_this_coord = np.sqrt(Ex[idx]**2 + Ey[idx]**2)
        if e_field_at_this_coord > max_E_field:
            max_E_field = e_field_at_this_coord
            idx_of_xcoord_at_max_E_field = (np.abs(x_uni-x)).argmin()
        else:
            pass
    return idx_of_xcoord_at_max_E_field 

# --- Determining depletion region edges and junction ---
x_p_depletion_edges_x = []  # Stores x-coords of left edges of depletion region
x_n_depletion_edges_x = []  # Stores x-coords of right edges of depletion region
x_p_depletion_edges_y = []  # Stores y-coords of left edges of depletion region
x_n_depletion_edges_y = []  # Stores y-coords of right edges of depletion region

idx_junc = idx_of_xcoord_at_max_field(y_slice)  # Index of the junction 

for j in y_uni:  # Iterates through every y-coordinate
    for i in x_uni[idx_junc:]:  # Searches from junction to the right of the device
        idx = lookup([i,j])
        e_field_at_this_coord = np.sqrt(Ex[idx]**2 + Ey[idx]**2)
        foundFlag = 0  # Flag to determine if depletion edge has been found
        if e_field_at_this_coord <= dep_edge_efield_mag:
            x_n_depletion_edges_x.append(i)
            x_n_depletion_edges_y.append(j)
            foundFlag = 1
            break
            
    if foundFlag==0:  # If threshold not encountered, assign x_device_right as right edge of dep region
        x_n_depletion_edges_x.append(x_device_right)
        x_n_depletion_edges_y.append(j)
        
    for i in np.flipud(x_uni[:idx_junc+1]):  # Searches from junction to the left of the device (Flipped array)
        idx=lookup([i,j])
        e_field_at_this_coord = np.sqrt(Ex[idx]**2 + Ey[idx]**2)
        foundFlag = 0  # Flag to determine if depletion edge has been found
        if e_field_at_this_coord <= dep_edge_efield_mag:
            x_p_depletion_edges_x.append(i)
            x_p_depletion_edges_y.append(j)
            foundFlag = 1
            break
            
    if foundFlag==0: # If threshold not encountered, assign x_device_left as left edge of dep region
        x_p_depletion_edges_x.append(x_device_left)
        x_p_depletion_edges_y.append(j)

# Turn the 4 lists into np arrays
x_n_depletion_edges_x = np.array(x_n_depletion_edges_x)
x_n_depletion_edges_y = np.array(x_n_depletion_edges_y)
x_p_depletion_edges_x = np.array(x_p_depletion_edges_x)
x_p_depletion_edges_y = np.array(x_p_depletion_edges_y)

def return_x_of_dep_edges(y):
    '''
    Determines the x-coordinates of both depletion edges at a given y-coordinate.
    Called in randomwalk diffusion and RPL outofbounds routines.
    
    Arguments
    ---------
    y : float in cm
    
    Returns
    -------
    X-coordinates of depletion edges : array 
        Array containing the x-coordinates of both depletion edges [cm] 
    '''
    idx = (np.abs(x_n_depletion_edges_y-y)).argmin()
    return [x_n_depletion_edges_x[idx],x_p_depletion_edges_x[idx]]

# Determining depletion width at y-slice
x_dep_edges_at_yslice = return_x_of_dep_edges(y_slice)
nominal_depletion_width = x_dep_edges_at_yslice[0] - x_dep_edges_at_yslice[1]

def escape_region(simulation_mode, coord):
    '''
    Determines if charge is still within bounds of the device. 
    Called whenever a charge changes its position to ensure it hasn't gone out of bounds. 
                  _____3____
            2    |          |   4
         ________|          |________
    1   |        .          .        |   5 
        |________.__________.________|
            8          7        6      
    
    Arguments
    ---------
    simulation_mode : 'RPL' or 'randomwalk'
        Affects the conditions for escape regions 1 and 5. 
    coord: array in the form [x,y] in cm
        Current position of the charge. 
    
    Returns
    -------
    code : int from 0 to 9. 
        0: Charge is still within bounds of the device. 
        1-8: Charge has escaped the device from the respective edges. 
    '''
    x, y = coord[0], coord[1]
    x_dep_edges_at_y = return_x_of_dep_edges(y)  # Depletion edges for current y-coord

    RPL = True if simulation_mode=='RPL' else False  # Are we in RPL mode?
    code = 0
    if (x < x_device_left ) or (RPL and (x < x_dep_edges_at_y[1])):  # If in p-neutral region during RPL, we consider it exiting from edge 1
        code = 1
    elif (x_device_left <= x <= x_core_left) and (y > wg_rib_y_coord):
        code = 2
    elif (x_core_left < x < x_core_right) and (y > y_core_top):
        code = 3
    elif (x_core_right <= x <= x_device_right) and (y > wg_rib_y_coord):
        code = 4
    elif (x > x_device_right) or (RPL and (x > x_dep_edges_at_y[0])):  # If in n-neutral region during RPL, we consider it exiting from edge 5
        code = 5
    elif (x_core_right <= x  <= x_device_right) and (y < y_core_bottom):
        code = 6
    elif (x_core_left < x < x_core_right) and (y < y_core_bottom): 
        code = 7 
    elif (x_device_left <= x <= x_core_left) and (y < y_core_bottom):
        code = 8
    
    return code
        
#As a sanity-check, print x-coords of the depletion edges at y-slice. 
print("")
print(f"Depletion region edges at y={y_slice*cm_to_nm:.1f} nm are x={x_dep_edges_at_yslice[1]*cm_to_nm:.3f} nm and x={x_dep_edges_at_yslice[0]*cm_to_nm:.3f} nm")
# -----------------------------------------------------------------------------------------------------------------------------

#################
## Random Walk ##
#################

# --- Randwalk parameters ---
core_height = abs(y_core_top - y_core_bottom)  # cm
rib_height = abs(wg_rib_y_coord-y_core_bottom)  # cm
if jnc_code == 'OV':  # If vertical junction is outside device
    width_of_n_neutral = abs(x_dep_edges_at_yslice[0]-x_device_right)  # cm
    stepsizelist_hole = [width_of_n_neutral,rib_height] 
elif jnc_code == 'IV':  # If vertical junction is inside device
    width_of_n_neutral = abs(x_core_right - x_dep_edges_at_yslice[0])  # cm
    stepsizelist_hole = [width_of_n_neutral,core_height] 
width_of_p_neutral = abs(x_core_left - x_dep_edges_at_yslice[1])  # The y-edges do not constitute a straight line, but a contour.
stepsizelist_electron = [width_of_p_neutral,core_height]

def rand_walk_params(stepsizelist):
    '''
    Determines random walk step size and max steps from the width of the neutral region and core height.
    
    Arguments
    ---------
    stepsizelist : array
        Array that contains [width of neutral region, height] in cm.
    
    Returns
    -------
    Random walk parameters : array
        Array that contains [random walk step size, random walk max steps]
    '''
    stepsizelist.sort() 
    stepsizeind = np.nonzero(stepsizelist)[0][0]  # Obtain index of smaller of the stepsizelist
    randwalk_stepsize = stepsizelist[stepsizeind] / randwalk_minsteps  # cm, eqn 2 in paper
    randwalk_std_timestep = ((randwalk_stepsize**2) / (2*20))  # s, eqn 3 in paper
    randwalk_maxsteps = randwalk_maxtime / randwalk_std_timestep
    return [randwalk_stepsize, randwalk_maxsteps]

# Print random walk parameters for sanity check 
print("")
print(f"Width of n-neutral region: {width_of_n_neutral*cm_to_nm:.2f} nm")
print(f"Width of p-neutral region: {width_of_p_neutral*cm_to_nm:.2f} nm")
print(f"Random-walk step size for electrons: {rand_walk_params(stepsizelist_electron)[0]*cm_to_nm:.2f} nm")
print(f"Random-walk step size for holes: {rand_walk_params(stepsizelist_hole)[0]*cm_to_nm:.2f} nm")

def diffusion(pos, region):
    '''
    Determines the diffusion parameters of a current position for a hole or electron in the neutral region. 
    
    Arguments
    ---------
    pos : array
        [x,y] coordinates in cm.
    region: int (0 or 1)
        0: electron in p region
        1: hole in n region
        
    Returns
    -------
    Tuple that contains (mobility, diffusion constant) in (cm^2/Vs, cm^2/s)
    '''
    idx = lookup(pos)
    mobility = mob_e[idx] if region == 0 else mob_h[idx]
    return mobility,mobility * (constants.k * temp / constants.e) 

# --- Main random walk function ---
d_list = []  # To store diffusion constants 
def rand_walk(start_pos,region):
    '''
    Random walk function. Performed when photon is injected into either the p or n neutral regions.
    The main loop iterates until the charge carrier fulfills an exit criteria (into the depletion region or out of the device),
    or reaches a pre-defined max number of steps.
    
    Arguments
    ---------
    start_pos : array
        The starting coordinates of the charge carrier in the form [x,y] in cm.
    region: int (0 or 1)
        0: electron in p region
        1: hole in n region
        
    Returns
    -------
    success : bool
        True if charge carrier diffuses into depletion region, else False
    total_time: float
        Time in seconds for random walk routine
    [xnew, ynew] : array
        New position of charge carrier after random walk in the form [x,y] in cm. 
    d_list : array
        List containing diffusion constants 
    exit_code : int 
        Result from calling escape_region. 
        0 : Charge is still within bounds of device 
        1-8: Charge carrier has escaped the device from the respective edges.
    '''
    # Initializing parameters 
    i = 0
    total_time = 0 
    pos = start_pos[:]
    success = False
    exit_code = 0
    if region == 0:  # If electron in p region
        randwalk_stepsize, randwalk_maxsteps = rand_walk_params(stepsizelist_electron)
    else:  # If hole in n region
        randwalk_stepsize, randwalk_maxsteps = rand_walk_params(stepsizelist_hole)
    
    # Main while loop
    while i < randwalk_maxsteps:
        # Parameters 
        D = diffusion(pos,region)[1]  # cm^2/s, Local diffusion constant
        current_time_step = (randwalk_stepsize ** 2) / (2*D)  # s, Time to travel randwalk_stepsize with local diffusion coefficient
        total_time += current_time_step  # s, Updates total time for random walk
        idx = lookup(pos) 
        efield_xy = [Ex[idx],Ey[idx]]  # [V/cm,V/cm], E-field in components [x,y]
        
        # Displacement due to drift
        drift_v_mag = driftv_e[idx] if region==0 else driftv_h[idx]  # cm/s, Drift velocity magnitude (Scalar)
        drift_v_mag_xy = drift_v_mag * np.fabs(efield_xy) / np.linalg.norm(efield_xy)  # [cm/s,cm/s], Drift velocity in components [x,y]
        sign = -1 if region == 0 else 1  # Sign = -1 for electrons in p region, +1 for holes in n region)
        drift_xy = sign * current_time_step * drift_v_mag_xy * np.sign(efield_xy)  # [cm, cm], Displacement due to drift in components [x,y]

        # Displacement due to brownian motion
        angle = random.uniform(0,2*math.pi)  # Brownian motion angle is randomly determined 
        brownian_xy = randwalk_stepsize * np.array([math.cos(angle),math.sin(angle)])  # [cm, cm], Displacement due to brownian motion in components [x,y]

        # Total random walk displacement
        [x_new, y_new] = pos  +  drift_xy + brownian_xy  # [cm, cm], Updates position of charge carrier
        
        # Check for bounds
        x_dep_edges_at_ynew = return_x_of_dep_edges(y_new)
        exit_code = escape_region('random_walk', [x_new,y_new])
        if exit_code != 0:  # Charge carrier has exited device 
            break
        elif region == 1 and x_new <= x_dep_edges_at_ynew[0]:  # n-depletion region is on the right
            success = True
            break
        elif region == 0 and x_new >= x_dep_edges_at_ynew[1]:  # p-depletion region is on the left
            success = True
            break
        
        # Update while loop parameters
        i += 1
        pos = [x_new, y_new]
        d_list.append(D)
       
    if not success:  # If charge carrier did not diffuse into depletion region, find out where it exited 
        exit_code = escape_region('random_walk', [x_new,y_new])
        
    return success, total_time, [x_new, y_new], d_list, exit_code
# -----------------------------------------------------------------------------------------------------------------------------

##############################
## Random Path Length (RPL) ##
##############################
# --- Class for individual charge carriers --- 
class Charge_carrier:
    _ids = count(0)  # To give each charge carrier a unique ID
    def __init__(self, name, pos, starttime, energy = 0.0, cum_prob = 0.0, cum_exponent = 0.0, current = 0.0):
        self.name = name  # 'hole' or 'electron'
        self.sign = 1 if name=='hole' else -1  # 1 for hole, -1 for electron
        self.pos = pos[:]  # [x,y] in cm, Current position of charge carrier
        self.random = random.uniform(0,1)  # -, Random number used in determining impact ionization
        self.energy = energy  # J, Energy of charge carrier
        self.cum_prob = cum_prob  # -, Cumulative probability of impact ionization
        self.cum_exponent = cum_exponent  # -, To calculate ionization cumulative density 
        self.current = current  # A,  Current contribution of charge carrier
        self.id = next(self._ids)  # -, Each charge carrier is given a unique ID 
        self.start_pos = pos[:]  # [x,y] in cm, Where charge is initialized
        self.start_t = starttime  # s, Time where charge carrier is initalized
        self.threshold_pos = None   # [x,y] in cm, Where charge carrier crosses threshold energy
        self.threshold_t= None  # s, Time where charge carrier crosses threshold energy
        self.impact_pos = None  # [x,y], Where charge carrier impact ionizes 
        self.impact_t = None  # s, Time where charge carrier impact ionizes
        self.travelled_distance = 0.0  # cm, To keep track of deadspace

# --- Calculation of ionization impact energies ---
def electron_impact(initial_energy): 
    '''
    Determines secondary electron and hole energies after electron impact ionization.
    
    Arguments
    ---------
    initial_energy : float
        Initial energy of electron in J.
        
    Returns
    -------
    Tuple that contains (secondary electron energy, secondary hole energy) both in J.
    '''
    energy_in_eV = initial_energy / constants.e
    secondary_electron_energy = 0.29 * energy_in_eV - 0.32  # in eV
    secondary_hole_energy = -1 * ( -0.31 * energy_in_eV - 0.92 ) - e_ionization_threshold  # in eV
    return secondary_electron_energy * constants.e, secondary_hole_energy * constants.e

def hole_impact(initial_energy):
    '''
    Determines secondary electron and hole energies after hole impact ionization.

    Arguments
    ---------
    initial_energy : float
        Initial energy of hole in J.

    Returns
    -------
    Tuple that contains (secondary electron energy, secondary hole energy) both in J.
    '''
    energy_in_eV = initial_energy / constants.e
    secondary_hole_energy = 0.375 * energy_in_eV - 0.476 # in eV
    secondary_electron_energy = -1 * ( -0.314 * energy_in_eV - 0.860 ) - h_ionization_threshold  # in eV
    return secondary_electron_energy * constants.e, secondary_hole_energy * constants.e    
    
# --- Main random path length function ---
def RPL(charge):
    '''
    Main RPL function that accepts a single charge carrier object, and simulates 1 RPL timestep.
    The properties of the charge carrier object (postiion, energy etc) are updated.
    
    Arguments
    ---------
    charge : object
        An instance of the charge_carrier class (electron or hole)
        
    Returns
    -------
    impact : bool
        True if charge successfully impact ionizes, else False.
    outofbounds: bool
        True if charge exits device, else False
    cross_ion_threshold : bool
        True if charge's energy exceeds ionization threshold, else False
    exit_code : int 
        Result from calling escape_region. 
        0 : Charge is still within bounds of device 
        1-8: Charge has escaped the device from the respective edges.
    '''
    # Initializing parameters
    impact = False
    outofbounds = False
    cross_ion_threshold = False
    idx = lookup(charge.pos) 
    efield_xy = [Ex[idx],Ey[idx]]  # [V/cm,V/cm], E-field in components [x,y]
    
    # Displacement due to drift 
    local_ion_coeff = ion_e[idx] if charge.name == 'electron' else ion_h[idx]  # -, Ionization coefficient
    drift_v_mag = driftv_e[idx] if charge.name == 'electron' else driftv_h[idx]  # cm/s, Drift velocity magnitude (Scalar)
    drift_v_mag_xy = drift_v_mag * np.fabs(efield_xy) / np.linalg.norm(efield_xy)  # [cm/s,cm/s], Drift velocity in components [x,y]
    
    # Update position of charge carrier
    [x_new,y_new] = charge.pos + RPL_timestep * charge.sign * drift_v_mag_xy * np.sign(efield_xy)  
    charge.pos = [x_new,y_new]  
        
    # Track the distance travelled by initial charge carriers to calculate deadspace
    if charge.start_t == 0:
        charge.travelled_distance += np.linalg.norm(drift_v_mag_xy) * RPL_timestep
    
    # Check for bounds
    exit_code = escape_region('RPL',[x_new,y_new])
    if exit_code != 0:  # Charge carrier has exited device
        outofbounds = True

    # Update charge attributes
    charge.energy += constants.e * np.inner(np.fabs(efield_xy), drift_v_mag_xy) * RPL_timestep  #J, Scalar, +ve regardless of e- or h+
    charge.current = -1*constants.e*drift_v_mag*np.inner(efield_xy/np.linalg.norm(efield_xy),[weight_Ex[idx],weight_Ey[idx]])  # A, via generalized Ramo's Theorem
    # Note: Drift v is a scalar, so we take velocity = drift v * unit vector of actual E-field to factor in direction. 
    # Electrons have a -1 factor from the negative charge (constants.e is +ve) and another -1 factor from moving against the E-field.
    # Thus, charge.current has an overall + sign. 

    # Check if ionization threshold has been reached
    ion_threshold = e_ionization_threshold if charge.name == 'electron' else h_ionization_threshold
    if charge.energy > ion_threshold:
        if charge.threshold_pos is None:  # If it is crossing threshold for the first time
            cross_ion_threshold = True
        else:
            charge.cum_exponent += local_ion_coeff * np.linalg.norm(drift_v_mag_xy) * RPL_timestep 
            charge.cum_prob += local_ion_coeff * math.exp(-1 * charge.cum_exponent) * np.linalg.norm(drift_v_mag_xy) * RPL_timestep  # Eqn 7 of 2018 paper
            if charge.cum_prob > charge.random:  # Check if probability exceeds random number generated in class definition
                impact = True

    return impact, outofbounds, cross_ion_threshold, exit_code
# -----------------------------------------------------------------------------------------------------------------------------

#############################
## Main Simulation Routine ##
#############################

# --- Initializing parameters ---
charge_list=[]  # For charge carrier objects
new_charge_list=[]
deadspace = None  # Distance travelled by charge in depletion region before ionizing
deadspace_start_coord = None
deadspace_end_coord = None
randwalk_success = False
randwalk_timetaken = 0
randwalk_finalcoord = []
exit_list = np.zeros(10, dtype=int)  # Counter to keep track of exit edges

# --- Generate injection point ---
injection_coord_raw = get_injection_point(pgen)
if test_suite:  # If in debugging mode, manually fix injection point
    injection_coord_raw = test_injection_coord
injection_coord_lookup_idx= lookup(injection_coord_raw)
injection_coord = [x[injection_coord_lookup_idx],y[injection_coord_lookup_idx]]  # [x,y] of injection point in [cm,cm]

# Print raw and mapped injection points
print("")
print(f"Raw injection point is [{injection_coord_raw[0]*cm_to_nm:.3f},{injection_coord_raw[1]*cm_to_nm:.3f}] nm")
print(f"Mapped injection point is [{injection_coord[0]*cm_to_nm:.3f},{injection_coord[1]*cm_to_nm:.3f}] nm")

# Sanity check, print depletion edges at injection point
x_dep_edges_at_inj_coord = return_x_of_dep_edges(injection_coord[1])
print(f"Depletion region edges at injection point are x={x_dep_edges_at_inj_coord[1]*cm_to_nm:.3f} nm and x={x_dep_edges_at_inj_coord[0]*cm_to_nm:.3f} nm")

# --- Check where does injection happen --- 

# Sanity check: does injection happen outside device?
# Should not be possible since injection point to the nearest coordinate within the device
if escape_region('random walk', injection_coord) != 0:
    region = - 2  
    print('The injection happens outside of the device!')

# Electron injection happens in p-neutral region. Begin random walk.
elif injection_coord[0] < x_dep_edges_at_inj_coord[1]:
    region = 0
    randwalk_success, randwalk_timetaken, randwalk_finalcoord, d_list, exit_code = rand_walk(injection_coord, region)
    if randwalk_success:  # If electron successfully diffuses into depletion region 
        charge_list.append(Charge_carrier('electron',randwalk_finalcoord, 0))
        print("")
        print(f"Injected one electron in p-neutral region. Time taken is {randwalk_timetaken*1e12:.3f} ps")
        print(f"Diffusion coordinate is [{randwalk_finalcoord[0]*1e7:.3f},{randwalk_finalcoord[0]*1e7:.3f}] nm")
    else:  # If electron does not diffuse into depletion region  
        print("")
        print(f"Injected one electron in p-neutral region, but it escaped after {randwalk_timetaken*1e12:.3f} ps at \
[{randwalk_finalcoord[0]*1e7:.3f},{randwalk_finalcoord[0]*1e7:.3f}] nm")
        exit_list[exit_code-1] += 1  # Record which edge it exited from 

# Hole injection happens in n-neutral region. Begin random walk. 
elif injection_coord[0] > x_dep_edges_at_inj_coord[0]:  
    region = 1
    randwalk_success, randwalk_timetaken, randwalk_finalcoord, d_list, exit_code = rand_walk(injection_coord, region)
    if randwalk_success:  # If hole successfully diffuses into depletion region
        charge_list.append(Charge_carrier('hole',randwalk_finalcoord, 0))
        print("")
        print(f"Injected one hole in n-neutral region. Time taken is {randwalk_timetaken*1e12:.3f} ps")
        print(f"Diffusion coordinate is [{randwalk_finalcoord[0]*1e7:.3f},{randwalk_finalcoord[0]*1e7:.3f}] nm")
    else:  # If hole does not diffuse into depletion region
        print("")
        print(f"Injected one hole in n-neutral region, but it escaped after {randwalk_timetaken*1e12:.3f} ps at \
[{randwalk_finalcoord[0]*1e7:.3f},{randwalk_finalcoord[0]*1e7:.3f}] nm")
        exit_list[exit_code-1] += 1  # Record which edge it exited from 

# Injection happens in the depletion region - Skip random walk
else:
    region = -1 
    charge_list.append(Charge_carrier('electron', injection_coord,0))
    charge_list.append(Charge_carrier('hole', injection_coord,0))
    print("")
    print("Injected one electron-hole pair in the depletion region")

# --- Print diffusion stats, if any ---
if d_list:
    mean = np.mean(d_list)
    print("")
    print(f"Mean diffusion constant: {mean:.2f} and std dev: {np.std(d_list):.2f}")
    randwalk_stepsize, randwalk_maxsteps = rand_walk_params(stepsizelist_hole) if region == 0 else rand_walk_params(stepsizelist_electron)
    print(f"The corresponding time step: {(((randwalk_stepsize ** 2) / (2*mean))* 1e15):.2f} fs")

# --- RPL routine in depletion region ---
#Initializing parameters
t = 0  # Not time, number of timesteps
max_timesteps = 1e-9 / RPL_timestep
avalanche = False   

while t < max_timesteps:  # Keep running until you've reached the max permitted timesteps
    current=0
    idx_to_be_removed=[]
    new_charge_list = []  # Created new list to avoid editting list as you iterate through it

    for i, charge in enumerate(charge_list):  # Loop over all existing charge carriers
        impact, outofbounds, cross_ion_threshold, exit_code = RPL(charge)

        if outofbounds:  # If charge is out of device, remove it and update exit code
            idx_to_be_removed.append(i)
            exit_list[exit_code-1] += 1
            continue

        if cross_ion_threshold:  # If charge crosses ionization threshold, update its attributes
            charge.threshold_pos = charge.pos[:]
            charge.threshold_t = t
            if deadspace is None and charge.start_t==0:
                deadspace = charge.travelled_distance
                deadspace_start_coord = charge.start_pos[:]
                deadspace_end_coord = charge.pos[:]
            
        if impact:  # If charge impact ionizes, update attributes and create new electron-hole pair
            charge.impact_pos = charge.pos[:]
            charge.impact_t = t
            if charge.name == 'electron':  # If charge is an electron
                new_electron_energy, new_hole_energy = electron_impact(charge.energy)
                charge.energy = new_electron_energy
            else:  # If charge is a hole 
                new_electron_energy, new_hole_energy = hole_impact(charge.energy)
                charge.energy = new_hole_energy

            # Append the new charge carriers into holding list
            new_charge_list.append(Charge_carrier('electron', charge.pos,t, energy=new_electron_energy)) 
            new_charge_list.append(Charge_carrier('hole', charge.pos,t, energy=new_hole_energy))
            # Reset these to 0
            charge.cum_prob = 0
            charge.cum_exponent = 0

        current += charge.current  # Add this charge's contribution to the total current 

        # Exits loop immediately without waiting for current time step to finish
        if current >= avalanche_threshold:
            avalanche = True
            break

    # Exits the bigger while loop too
    if avalanche == True:  
        break

    # After 1 time step, removed unwanted charges and add new charges 
    for i in sorted(idx_to_be_removed, reverse=True):  # Must delete in reverse order to not mess up the index!
        del charge_list[i]
    charge_list += new_charge_list

    if (t%250)==0:  # Prints updates at specific time intervals 
        print(f"Timesteps taken = {t}, current = {current*1e3:.4f} mA, Number of charges = {len(charge_list)}")
              
    t += 1  # Move on to the next timestep 

    if not charge_list:  # If there are no charge carriers left, end RPL
        break 

# --- Final outputs & cleanup ---
if avalanche:
    print("")
    print(f"Avalanche was successful!")
    print(f"Total time taken: {(t*RPL_timestep+randwalk_timetaken)*1e12:.3f} ps, Timesteps taken: {t}")
    print(f"Diffusion time taken: {randwalk_timetaken*1e12:.3f} ps")
    print(f"Final current: {current*1e3:.6f} mA, Final number of charges: {len(charge_list)}")
else:
    print("")
    print(f"Avalanche failed!")
    print(f"Total time taken: {(t*RPL_timestep+randwalk_timetaken)*1e12:.3f} ps, Timesteps taken: {t}")
    print(f"Diffusion time taken: {randwalk_timetaken*1e12:.3f} ps")
    print(f"Final current: {current*1e3:.6f} mA, Final number of charges: {len(charge_list)}")
    print("")
    print(f"Charges escaped x times from the following edges:")
    print("Edges: [1,2,3,4,5,6,7,8,9,10]")
    print(f"Exits: {exit_list}")

if deadspace is not None:  # Print deadspace stats, if any
    print("")
    print(f'Initial charge carrier crossed ionization energy threshold. Dead space travelled is: {deadspace*1e7:.3f} nm')
    print(f'Start pos: [{deadspace_start_coord[0]*1e7:.3f},{deadspace_start_coord[1]*1e7:.3f}] nm, \
crossed threshold at: [{deadspace_end_coord[0]*1e7:.3f},{deadspace_end_coord[1]*1e7:.3f}] nm')

elapsed_time = time.time() - start_time 
print('Elapsed time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
# -----------------------------------------------------------------------------------------------------------------------------

##############
## Plotting ##
##############
# Plots the device's edges, both depletion edges, the injection coordinate and the 
              
if test_suite:    
    print("The height of n- and p-neutral layers are: {:.0f} and {:.0f} nm, respectively.".format(width_of_n_neutral*1e7,width_of_p_neutral*1e7))
    plt.plot(x_p_depletion_edges_x*cm_to_nm,x_p_depletion_edges_y*cm_to_nm, color='r')  # Plots the p-depletion edge 
    plt.plot(x_n_depletion_edges_x*cm_to_nm,x_n_depletion_edges_y*cm_to_nm)  # Plots the n-depletion edge
    plt.plot(injection_coord[0]*cm_to_nm,injection_coord[1]*cm_to_nm, marker='o', markersize=3, color="red")  # Plots the injection coordinate 
    
    if randwalk_finalcoord:  # Plots the random walk final coordinate, if random walk was called 
        plt.plot(randwalk_finalcoord[0]*cm_to_nm,randwalk_finalcoord[1]*cm_to_nm, marker='o', markersize=3, color="blue")
    
    plt.xlim(x_device_left*cm_to_nm,x_device_right*cm_to_nm)
    plt.ylim([y_core_bottom*cm_to_nm,y_core_top*cm_to_nm])
    
    # --- Plot device edges ---
    device_x_edges = [x_core_left*cm_to_nm,x_core_right*cm_to_nm]
    for i in device_x_edges:
        plt.axvline(i,ymin=(wg_rib_y_coord-y_core_bottom)/(y_core_top-y_core_bottom), ymax=1,color='k')
    plt.axhline(wg_rib_y_coord*1e7,xmin=0.0, xmax=abs((x_device_left-x_core_left)/(x_device_left-x_device_right)),color='k')
    plt.axhline(wg_rib_y_coord*1e7,xmin=abs((x_device_left-x_core_right)/(x_device_left-x_device_right)), xmax=1,color='k')    
    
    plt.yticks([y_core_bottom*cm_to_nm,0,y_core_top*cm_to_nm])
    plt.show()
