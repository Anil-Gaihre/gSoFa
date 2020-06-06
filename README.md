Software (tested)
-----
g++ 5.4.0, g++ 7.5.0, IBM Spectrum MPI 10.3.0.0, CUDA 10.0
Compilation flag: -O3

--
Hardware (tested)
------
V100 (tested)

--
Compile
-----

make

--
Execute
------
Test (Runs in BC dataset): make test
Type: "./mpi_cuda_test.bin" it will show you what is needed.
Tips: The test graph (PR in the paper) is in the Pre2 folder.

--
Dataset generation
----
We use SuperLU_DIST (https://github.com/xiaoyeli/superlu_dist) to get the exact reordered matrix that it uses for the Symbolic Factorization for fair comparison.

The reordered matrix by the SuperLU_DIST solver before symbolic factorization in it is given as an input to gSoFa.
We dump the reordered matrix from SuperLU_DIST solver to CSR files. An example dataset in the pre2 folder (PR) is in ascii format in dataset folder.

--
Code specification
---------

Without_space_optimization:
Consists of the code that doesn't do space optimization. 
Compile: make
Run: make test
mpiexec -n 1 $(exe) begin_position.dat end_position.dat csr.dat N_GPU 1024 0 0 1 1 4096

With_space_optimization:
Consists of the code that includes the space optimizations. 
Compile: make
Run: make test
mpiexec -n 1 $(exe) begin_position.dat end_position.dat csr.dat N_GPU 1024 0 0 1 1 4096


The overall code structure in each folder is:

- main.cu: main function.
-symbfact.cuh: computes symbolic factorization
-barrier.cuh: Consists of software barrier
- wtime.h: timing.

Note: The code is modified to run on a single GPU. 
The process don't communicate during the run-time, so reviewer can increase the num_process parameter in command line to get the performance of multiple GPUs virtually by a single GPU.
