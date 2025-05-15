# Interplay of metallicity, ferroelectricity and layer charges in SmNiO3/BaTiO3 superlattices

This repository contains all code needed to reproduce data analysis and figures in the [publication](https://doi.org/10.1103/PhysRevResearch.7.023044) by Edith Simmen and Nicola A. Spaldin. The data needed to run the code available in the following [repository](https://doi.org/10.24435/materialscloud:gy-nd).

This repository contains the following files: 
- The notebook "paper_figures.ipynb" reads in all the data and generates the figures in the same order as shown in the paper. 
- The python notebook "elstat_model_best_parameters.ipynb" uses the different electrostatic models (with or without additional screening by electron-hole excitation) to find the parameters that best reproduce DFT behavior and also generates the data plotted in the paper. 
- The electrostatic models and all additional functions needed to process the data are found in the python_functions directory.
- The required packages and the package versions used to create these notebooks are given in the requirements file. 

