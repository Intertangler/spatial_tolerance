# Antibody binding stochastic simulation

This repository contains Python libraries and jupyter notebooks for simulating and analyzing patterned surface plasmon resonance (PSPR) data and studying the spatial tolerance phenomenon of bivalent antibodies and patterned antigen substrates. The code can be run by setting up the appropriate Python environment. For information about the math and physics behind the code, see our paper:  Hoffecker IT, Shaw A, Sorokina V, Smyrlaki I, Högberg B. The geometric determinants of programmed antibody migration and binding on multi-antigen substrates. bioRxiv. 2020 Jan 1. https://doi.org/10.1101/2020.10.12.336164

The code is designed to run on linux, though it may work on other operating systems if the appropriate Python environment is installed. 

# Installation instructions

* Go to https://github.com/Intertangler/spatial_tolerance.git and download the whole folder.
* Extract the contents to a folder.
* Create a Python 2.7 conda environment: conda create -n 210805codetest2 python=2.7.17
* Navigate to the folder containing ST_environment.yml and ST_environment_xplatform.yml files
* You may need to rename the "prefix" at the end of the yml file to specify the appropriate local folder: /home/user/miniconda3/envs/STenv
* Create an environment called STenv with:     conda env create -f spatial_tolerance_environment.yml if you are on Linux or ST_environment_xplatform.yml if you are on another OS.
* Navigate to the main spatial tolerance directory and start jupyter lab with the command: jupyter lab
* From jupyter lab, navigate into each subfolder and start a notebook by doubleclicking.
* The notebooks demonstrate the workflows used to generate the results and figures from the paper.
