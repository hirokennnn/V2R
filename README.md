# V2R

This repository contains codes and data for the following paper:   
> Mathematical Framework to Identify Optimal Molecule based on Virtual Ligand Strategy.
> Wataru Matsuoka, Ken Hirose, Ren Yamada, Taihei Oki, Satoru Iwata, Satoshi Maeda, _ChemRxiv_ **2025**, 
> DOI: ([10.26434/chemrxiv-2025-0snj5](https://chemrxiv.org/engage/chemrxiv/article-details/67d8c3fa6dde43c90830259a))  

If codes and/or data in this repository are used in your work, please cite the following papers:
> _ACS Catal._ **2024**, _14_, 16297.  
> _ChemRxiv_ **2025**, DOI: 10.26434/chemrxiv-2025-0snj5. 

It consists of three main components:
- VL : python programs to calculate the penarty, gradient and Hessian for the virtual ligand method
- optimizer : python programs to perform the VLAO calculation
- data: representative input files and data sets
  
Each component has a readme file with further information.  
The codes have been slightly modified from the version used in the original paper, but it has been confirmed that essentially the same results are obtained.   
The codes have only been tested in the computer environment below, and may require minor modifications to run in different computer systems.  

## Test environment
- GRRM23 program  
- Gaussian 16
- conda 23.7.4
- Python 3.11.5
- pytorch 2.1.0  
- numpy 1.24.3  

