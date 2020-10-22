# BootCMatchG
Sparse solvers are one of the building blocks of any technology for reliable and high-performance scientific and engineering computing. In BootCMatchG we make available a recently proposed adaptive Algebraic MultiGrid (alpha-AMG) method for preconditioning algebraic linear systems Ax = b, where A is a symmetric positive definite (s.p.d.), large and sparse matrix. All the computational kernels for setup and application of the adaptive AMG method, as preconditioner of an efficient version of the Conjugate Gradient Krylov solver, were designed and tuned for accessing GPU global memory according to best practices of CUDA programming and for using the available computing resources in an effective way. To the best of our knowledge, this is the only adaptive AMG method currently available as public-domain software package and running on GPU.

This is the software package described in the paper:
M. Bernaschi, P. D’Ambra, D. Pasquini, AMG based on compatible weighted matching on GPUs, Parallel Computing. Vol. 92, 2020. https://doi.org/10.1016/j.parco.2019.102599

This project was partially supported by the EC under the Horizon 2020 Project: Energy Oriented Center of Excellence (EoCoE II): Toward Exascale
for Energy, Project ID: 824158.


## Installation
### Dependencies and Requirements:

The software requires:
* CUDA >= 9.0
* **[CUB](https://nvlabs.github.io/cub/)**: we tested the code for the version: 1.7.4.", but any newer version should work as well.
  * Install the software and setup the variable **CUB_PATH** in the BootCMatchG's Makefile: e.g., CUB_PATH = ./cub-1.7.4
* **[NSPARSE](https://github.com/EBD-CREST/nsparse)**: We included in this repository a slightly modified version *NSPARSE* that supports CUDA > 9.0. This is located in *src/EXTERNAL/*

### Compilation

> cd src; make

## Solving 

The file *solve_example.cu* gives an example of how to use BootCMatchG. It takes two inputs:

> ./bin/solve_example [INPUT_MATRIX] [AMG_setting] 

*INPUT_MATRIX* is a sparse matrix in the *Matrix Market* format and *AMG_setting* is a configuration file (more information on that later).

*solve_example.cu* solves the system using a unitary right-hand side and prints the solution in *./x.txt*.

An example:

> ./bin/solve_example ../test_matrix/pi8grid6.mtx ../AMGsettings_base

### Configuration file

The configuration file defines the preconditioning and solving procedure.

The are 15 parameters:

* solver_type: type of final AMG composition; 0 multiplicative, 1 symmetrized multi., 2 additive; NB: Here put 0 for single AMG component
* max_hrc: max number of hierarchies in the final bootstrap AMG; NB: Here put 1 for single AMG component
* rho: desired convergence rate of the composite AMG; NB: This is not generally obtained if criterion on max_hrc is reached
* aggrsweeps: pairwise aggregation steps; 1 for single step, 2 for pairs; 3 for double pairs ...
* aggr_type: type of prolongation; 0 unsmoothed, 1 smoothed
* max_levels: max number of levels built for the single hierarchy
* cycle_type: 0-Vcycle, 1-Hcycle (V-cycle at odd levels and W at even levels), 2-Wcycle
* coarsest_solver_type: 0 Jacobi, 4 L1-smoother
* relax_type: 0 Jacobi, 4 L1-smoother
* relaxnumber_coarse: number of iterations for the coarsest solver
* prerelax_sweeps: number of pre-smoother iterations at the intermediate levels
* postrelax_sweeps: number of post-smoother iterations at the intermediate levels
* itnlim: maximum number of iterations for the solver
* rtol: relative accuracy on the solution

An example of configuration file is given in */AMGsettings_base*

---
How to cite our work:
> @article{BERNASCHI2020102599, <br>
title = "AMG based on compatible weighted matching for GPUs",<br>
journal = "Parallel Computing",<br>
volume = "92",<br>
pages = "102599",<br>
year = "2020",<br>
issn = "0167-8191",<br>
author = "Massimo Bernaschi and Pasqua D’Ambra and Dario Pasquini",<br>
keywords = "AMG, Graph matching, GPU",<br>
}
