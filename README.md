# Antibody binding stochastic simulation

## Overview

This repository contains Python libraries and jupyter notebooks for simulating and analyzing patterned surface plasmon resonance (PSPR) data and studying the spatial tolerance phenomenon of bivalent antibodies and patterned antigen substrates. The code can be run by setting up the appropriate Python environment. For information about the math and physics behind the code, see our paper:  Hoffecker IT, Shaw A, Sorokina V, Smyrlaki I, Högberg B. The geometric determinants of programmed antibody migration and binding on multi-antigen substrates. bioRxiv. 2020 Jan 1. https://doi.org/10.1101/2020.10.12.336164

The code is designed to run on linux, though it may work on other operating systems if the appropriate Python environment is installed. 

## System requirements
### Dependencies:
#### Software: ####
- Python 2.7
- Jupyter notebook / Jupyter lab
- Anaconda 

#### Operating system: ####
- Ubuntu 18.04.05
- macOS Mojave 10.14.6
 
#### Python dependencies: ####
  - python=2.7
  - jupyterlab
  - matplotlib
  - numpy
  - scipy
  - ipdb
  - networkx
  - pandas
  - tqdm
  - python-levenshtein
  - biopython
  - sympy
  - svgwrite


## Installation instructions

* Download the contents of this git repository (https://github.com/Intertangler/spatial_tolerance.git).
* Extract the contents to a folder.
* Navigate to the folder containing ST_environment.yml and ST_environment_xplatform.yml files
* You may need to rename the "prefix" at the end of the yml file to specify the appropriate local folder: /home/user/miniconda3/envs/STenv
* Create an environment called STenv with:     conda env create -f spatial_tolerance_environment.yml if you are on Linux or ST_environment_xplatform.yml if you are on another OS.
* Navigate to the main spatial tolerance directory and start jupyter notebook with the command: jupyter notebook
* From jupyter, navigate into each subfolder and start a notebook by doubleclicking.
* The notebooks demonstrate the workflows used to generate the results and figures from the paper.
* If you have problems setting up the environment this way, you may try to obtain the packages individually:
* The typical time it takes to install all dependencies is 10-20 minutes
  
  
## Demo
To run our code, use the Jupyter notebooks provided which call functions from the respective libraries we have developed and which generate data, analyses, and figures such as those used in the manuscript.
### Basic demo
- Complete the basic demo by opening the folder _0_user_demo and running the CTMC_demo.ipynb jupyternotebook: https://github.com/Intertangler/spatial_tolerance/blob/master/_0_user_demo/CTMC_demo.ipynb
- Follow the instructions in the notebook to create a transition matrix with user-defined antigen coordinates, simulate antibody dynamics with the enumerative/deterministic CTMC method, as well as simulate using the sampling-based Monte Carlo approach.
### Instructions to run on data
#### Generating transition matrices
- To generate the transition matrices including those featured in the publication or transition matrices for custom antigen patterns, go to folder 000_generate_transition_matrices and open the Jupyter notebook: https://github.com/Intertangler/spatial_tolerance/blob/master/000_generate_transition_matrices/generate_new_transition_matrix.ipynb 
- Complete the input data including coordinates and cutoff distance in the cell displayed in the notebook and run the cell to generate matrices
- The output files should appear in the folder transition_matrices and should include .npy files and blank text files corresponding to occupancy_key, particle_count, and transition_matrix files.
- The expected runtime depends on the complexity and connectivity of the input antigen pattern. For not-fully-connected patterns with less than 6 antigens, run times may take seconds to a few minutes. Highly-connected patterns with 6 or more antigens may take several minutes to hours to find all states. Patterns with more than 12 antigens may not be realistically completed in time, and a Monte Carlo approach should be used instead - see manuscript.
#### Constructing a model from SPR data
- To parameterize a Markov model from SPR data, navigate to folder 001_progressive_fitting_alpha_energies_crossval and open the Jupyter notebook https://github.com/Intertangler/spatial_tolerance/blob/master/001_progressive_fitting_alpha_energies_crossval/fitting_master_igg1dig_junedata_apparent.ipynb
- Run all cells in the notebook to perform fitting and parameterization according to IgG1 anti-digoxygenin data from the manuscript or substitute your own SPR data substitute the file names at the top of the notebook.
- The output you should get is a series of figures including raw/fitted SPR data plots, stratified probability versus time plots, stratified occupancy plots, and a parameterized spatial tolerance function and plot of fit.
- The expected run time is less than 30 minutes for a laptop computer or average desktop PC. 

### Instructions to reproduce manuscript results
To reproduce the manuscript results, run each notebook without modifying any inputs.
  - https://github.com/Intertangler/spatial_tolerance/blob/master/000_generate_transition_matrices/generate_new_transition_matrix.ipynb
  - https://github.com/Intertangler/spatial_tolerance/blob/master/001_progressive_fitting_alpha_energies_crossval/fitting_master_igg1dig_junedata_apparent.ipynb
  - https://github.com/Intertangler/spatial_tolerance/blob/master/002_progressive_fitting_JR_cross_val/cross_validation_june.ipynb
  - https://github.com/Intertangler/spatial_tolerance/blob/master/003_visual_ctmc_w_energy_analysis/general_CTMC_IgG1_dig.ipynb
  - https://github.com/Intertangler/spatial_tolerance/blob/master/004_monte_carlo1/mcmc_run.ipynb

## License
The software in this repository is licensed under the GNU General Public License (GPL).

## Description of code
Description of the code, the underlying mathematics, and its application can be found in the Methods section of our manuscript. Hoffecker IT, Shaw A, Sorokina V, Smyrlaki I, Högberg B. The geometric determinants of programmed antibody migration and binding on multi-antigen substrates. bioRxiv. 2020 Jan 1. https://doi.org/10.1101/2020.10.12.336164
