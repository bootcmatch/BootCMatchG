# BootCMatchGPU
... small intro ....
## Installation
### Dependencies and Requirements:

The software requires:

* **[CUB](https://nvlabs.github.io/cub/)**: We test the code for the version: 1.7.4.
  * Install the software and setup the variable **CUB_PATH** in the BootCMatchG's Makefile: e.g., CUB_PATH = ./cub-1.7.4
* **[NSPARSE](https://github.com/EBD-CREST/nsparse)**: Install the software and setup the variable **NSPARSE_PATH** in the BootCMatchG's Makefile: e.g., NSPARSE_PATH = ./nsparse-master/cuda-c

### Compilation

> cd src; make

## Solving 

The file *solve_example.cu* gives an example of how to use BootCMatchGPU. It takes two inputs:

> ./src/bin/solve_example [INPUT_MATRIX] [AMG_setting] 

*INPUT_MATRIX* is a sparse matrix in the *Matrix Market* format and *AMG_setting* is a configuration file (more information on that later).

*solve_example.cu* solves the system using a unitary right-hand side and prints the solution in *./x.txt*.

An example:

> ./src/bin/solve_example ../test_matrix/pi8grid6.mtx ../AMGsettings_base

### Configuration file

The configuration file defines the preconditioning and solving procedure.

The are 15 parameters:

* solver_type: 0 multiplicative, 1 symmetrized multi., 2 additive; NB: Here put 0 for single AMG component
* max_hrc, in bootstrap AMG, max hierarchies; NB: Here put 1 for single AMG component
* desired convergence rate of the composite AMG; NB: This is not generally obtained if criterion on max_hrc is reached
* matchtype: 3 Suitor
* aggrsweeps; passi di aggregazione a coppie. 0, pairs; 1 double pairs ...
* aggr_type; 0 unsmoothed, 1 smoothed
* max_levels; max number of levels built for the single hierarchy
* cycle_type: 0-Vcycle, 1-Hcycle, 2-Wcycle
* coarsest_solver_type: 0 Jacobi, 4 L1-smoother
* relax_type: 0 Jacobi, 4 L1-smoother
* relaxnumber_coarse
* prerelax_sweeps
* postrelax_sweeps
* itnlim
* rtol

An example of configuration file is given in *./AMGsettings_base*



