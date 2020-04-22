Dataset: BC dataset in the paper

Generation of re-ordered CSR: The reordered matrix by the SuperLU_DIST solver before symbolic factorization is taken from the SuperLU_DIST solver as an input to gSoFa.
We dump the reordered matrix from SuperLU_DIST solver to CSR files in dataset folder.

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

Note: The code is modified to run on a single GPU. 
The process don't communicate during the run-time, so reviewer can increase the num_process parameter in command line to get the performance of multiple GPUs virtually by a single GPU.
