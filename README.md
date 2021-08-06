# Antibody binding stochastic simulation

## Overview

This repository contains Python libraries and jupyter notebooks for simulating and analyzing patterned surface plasmon resonance (PSPR) data and studying the spatial tolerance phenomenon of bivalent antibodies and patterned antigen substrates. The code can be run by setting up the appropriate Python environment. For information about the math and physics behind the code, see our paper:  Hoffecker IT, Shaw A, Sorokina V, Smyrlaki I, Högberg B. The geometric determinants of programmed antibody migration and binding on multi-antigen substrates. bioRxiv. 2020 Jan 1. https://doi.org/10.1101/2020.10.12.336164

The code is designed to run on linux, though it may work on other operating systems if the appropriate Python environment is installed. 

## System requirements
### Software dependencies and operating systems:
'''Software has been tested on:'''
 Ubuntu 18.04.05
 macOS Mojave	10.14.6
 
'''Python dependencies:'''
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
 


# Installation instructions

* Download the contents of this git repository (https://github.com/Intertangler/spatial_tolerance.git).
* Extract the contents to a folder.
* Navigate to the folder containing ST_environment.yml and ST_environment_xplatform.yml files
* You may need to rename the "prefix" at the end of the yml file to specify the appropriate local folder: /home/user/miniconda3/envs/STenv
* Create an environment called STenv with:     conda env create -f spatial_tolerance_environment.yml if you are on Linux or ST_environment_xplatform.yml if you are on another OS.
* Navigate to the main spatial tolerance directory and start jupyter notebook with the command: jupyter notebook
* From jupyter, navigate into each subfolder and start a notebook by doubleclicking.
* The notebooks demonstrate the workflows used to generate the results and figures from the paper.
* If you have problems setting up the environment this way, you may try to obtain the packages individually:


  
  



