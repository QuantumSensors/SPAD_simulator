## Introduction
This is a 2D Monte Carlo simulator written in Python to model the operation of single-photon avalanche detectors. It simulates the stochastic avalanche multiplication process of charge carriers following the absorption of an input photon; a successful detection event is defined as the avalanche current exceeding a pre-defined threshold. The simulator output can be used to analyse the photon detection efficiency and its timing characteristics. This program does not directly simulate dark noise.

The code was originally written to simulate p-n or p-i-n type silicon devices with a rib waveguide geometry, but can be generally extended to different materials or geometries. Further details can be found in [\[1-2\]](#references); the code presented here reflects the same version used in the more recent paper [\[2\]](#references).

Original authors: Salih Yanikgonul, Victor Leong, Junrong Ong\
at the Agency for Science, Technology, and Research (A*STAR), Singapore\
Additional contributions by Gundlapalli Prithvi and Dillon Lim

## Required Inputs

The simulator requires three input files. Refer to the provided examples of these input files in this repository for the required format. The example files reflect an actual device that simulated and reported in [\[2\]](#references).

##### 1) Data_file: DC electrical characteristics 
A table containing electric field and related data covering the simulation domain. Required quantities:
- x,y coordinates 
- Electric field (x,y)
- Weighting field (x,y) 
- Electron and hole ioniziation coefficients 
- Electron and hole mobilities
- Electron and hole drift velocities

The weighting field is the electric field obtained under these modified conditions: (i) the cathode is at unit potential, while the anode is grounded; (ii) all charges (including space charges) are removed, i.e. the waveguide is undoped. More details can be found in [\[2\]](#references).

##### 2) Pgen_file: Distribution of injection coordinates (i.e. where the photon is absorbed)
Note that the name "pgen" used in the code refers to the inital _photogenerated_ charge carriers arising from a photon absorption.

In most cases, the spatial distribution of injection coordinates should reflect the optical mode within the waveguide, but this can be modified to suit different test scenarios.

The required data format is in the form of a cumulative probability table. To obtain this from the optical waveguide mode, the field data can integrated over all the coordinates and then normalized. This should result in a table where each (x,y) coordinate is associated with a cumulative probability from 0 to 1. 

To determine the injection point in each run, the simulator generates a random number from 0 to 1, and then chooses the coordinate in the table that (most closely) matches that number.


##### 3) Parameter_file: device geometry, thresholds, and other constants. 
Other simulation parameters including:
- Temperature
- Threshold electric field to determine the depletion region 
- Avalanche current threshold to determine a successful avalanche
- Parameters for random walk random path length (RPL) models
- Defining the edge coordinates of the waveguide -- the simulator disregards charge carriers that exit the waveguide
- Position of diode junction (jnc_code)
    - jnc_code is either 'IV' for a vertical junction that is inside the rib, and 'OV' for a vertical junction that is outside the rib.
- The junction code affects the definition of the n-neutral region in the random walk section of the code. 

<p align="center">
    <img src="/images/jnc_comparison.jpg", width="1400", height="549">
    <br> <i> Figure 1: Comparison between inside junction and outside junction.</i>
</p>

- The set of data files provided in this folder are for a device of with the following specifications:
<br> Width 900nm, height 340nm, rib height 270nm, junction displacement +300nm, intrinstic region width 300nm, as depicted in figure 1d of the research paper (linked below) with a bias voltage of 32.2V applied.
<p align="center">
    <img src="/images/device_schematic.jpg">
    <br> <i> Figure 2: Schematic depicting defined edges on the device.</i>
</p>
y_slice is a nominal y-coordinate used to determine the depletion regions. It should be somewhere in the middle of the rib portion of the waveguide.

## Code structure
The code is segmented into 9 main segments: 
1. Imports - Import various libraries needed 
2. Debugging - Variables and functions used for debugging
3. Reading input files - Parses and reads the 3 data files
4. Variables - Physical constants, info from the data files, definition of charge carrier class
5. Functions - Misc functions and preliminary routines
6. Random Walk - Random walk routine if injection happens in neutral regions
7. Random Path Length - RPL routine for charge carriers in depletion region 
8. Main simulation routine - Generates injection point. For each time step, calls randwalk (if required) and RPL routines over all charge carriers, until success/failure condition is reached
9. Plotting - Matplotlib settings for plotting the device edges, depletion edges, injection point and random walk final point. Only runs if test_suite is set to True. 

## Main Simulation Sequence 
#### 1) Injection Point Generation
- Injection point is generated randomly via get_injection_point().
- The injection coordinate can be classified into different regions (see Fig 2) 
    - (i) If injection point falls outside of device, avalanche fails and the simulation is over. 
    - (ii) If injection point falls in the p or n neutral regions,the random walk routine is called.
    - (iii) If injection point falls in the depletion region, the random path length (RPL) routine is called.

#### 2) Random Walk
- Called only when injection point falls in the p or n neutral regions. 
- At each timestep, the displacement of the charge carrier is due to a vector sum of its drift velocity and brownian motion. 
- If the charge carrier enters the depletion region, the RPL routine is called. 
- If the charge carrier does not reach the depletion region by the maximum number of steps permitted, the avalanche fails and the simulation is over. 

#### 3) Random Path Length (RPL)
- This routine is iterated over all individual charge carriers.
- The charge carriers are accelerated by the electric field.
- Avalanche multiplication, i.e. the generation of additional charge carriers, happens probabilistically when the charge carriers have sufficient energy.
- The avalanche is successful when the device current exceeds the avalanche threshold.
- If the current does not exceed the avalanche threshold by the maximum time permitted, or if the number of charges falls to 0, then the avalanche is a failure and the simulation is over. 
<p align="center">
    <img src="/images/outcome_tree.jpg">
    <br> <i> Figure 3: Tree diagram depicting possible outcomes in flow of code.</i>
</p>

## Execution
- Sample execution of code from the command line: 
```
python3 SPAD_simulation.py -data_file data.out -pgen_file pgen.dat -parameter_file parameters.dat -run_number 1
```
- This parses in the 3 data files and a run number for documentation purposes. 
- Sample outputs from the code: 

<p align="center">
    <img src="/images/successful_avalanche.jpg", width="660", height="582">
    <br> <i> Figure 4: Sample output for successful avalanche. </i>
    <br>
    <br><img src="/images/unsuccessful_avalanche.jpg", width="660", height="516">
    <br> <i> Figure 5: Sample output for unsuccessful avalanche. </i>
</p>

## Test Suite 
- If the variable test_suite is set to True, certain changes will be made to the code for the purpose of debugging. 
1) The input files will be hard coded, thus you no longer need to include them as arguments when executing the code. It can now be simply executed as by:
```
python3 SPAD_simulation.py
```
2) The injection point will be fixed to a point test_injection_point, which can be set in the debugging section of the code. 
3) Several plots will be generated, including the device's edges, its depletion edges, the injection coordinate and the random walk final coordinate (if any) will be displayed. 
<p align="center">
    <img src="/images/sample_plot.jpg", width="550", height="425">
    <br> <i> Figure 6: Sample test_suite plot.</i>
</p>

## References
\[1\] [S. Yanikgonul, V. Leong, J. R. Ong, C. E. Png, and L. Krivitsky, "2D Monte Carlo simulation of a silicon waveguide-based single-photon avalanche diode for visible wavelengths," Opt. Express 26, 15232-15246 (2018).](https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-26-12-15232&id=390108)

\[2\] [S. Yanikgonul, V. Leong, J. R. Ong, C. E. Png and L. Krivitsky, "Simulation of Silicon Waveguide Single-Photon Avalanche Detectors for Integrated Quantum Photonics," IEEE Journal of Selected Topics in Quantum Electronics 26, 1-8 (2020)](https://ieeexplore.ieee.org/document/8820038)