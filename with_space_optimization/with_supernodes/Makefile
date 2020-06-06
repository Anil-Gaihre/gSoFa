exe=./mpi_cuda_test.bin
deb=./exe_debug
N_Process=1
#mp=/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/xl-16.1.1-3/spectrum-mpi-10.3.0.0-20190419-q75ow22tialepxlgxcs7esrptqwoheij/bin/mpirun
p0=0
p1=0
b=1
cucc= "$(shell which nvcc)" 
cc= "$(shell which mpicxx)"
g="$(shell which g++)"
commflags=-lcudart -L"$(shell dirname $(cucc))"/../lib64
deb_flag=-g -G
#cuflags= --cempiler-options -Wall -Xptxas -v -Xcudafe -\# --reseurce-usage 
  cuflags= --compiler-options -Wall -v -Xcudafe -\# --resource-usage  
#cuflags= --compiler-options -Wall -Xptxas -dlcm=cg ## disable L1 cache
objs	= $(patsubst %.cu,%.o,$(wildcard *.cu)) \
	$(patsubst %.cpp,%.o,$(wildcard *.cpp))

deps	= $(wildcard ./*.cuh) \

#disL1 = –Xptxas -v –dlcm=cg
%.o:%.cu $(deps)
	$(cucc) -c $(cuflags) $< -o $@

%.o:%.cpp $(deps)
	$(cc) -c  $< -o $@

$(exe):$(objs)
#	$(cc) $(objs) $(commflags) -o $(exe)
	 $(cc) $(objs) $(commflags) -o $(exe) 

# $(deb):$(objs)
 #	$(cc) $(objs) $(commflags) -o $(exe)
# 	 	$(g) $(objs) $(commflags) -o $(deb) $(deb_flag)
test_memcheck:$(exe)
		mpiexec -n 1 cuda-memcheck --leak-check full --error-exitcode 1 $(exe) /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_11948.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_11948.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_11948.dat 1 32 $(p0) $(p1) $b 6

local:$(exe)
		mpiexec -n 1 $(exe) ../colbeg_11948.dat ../colend_11948.dat ../csr_11948.dat 1 128 $(p0) $(p1) $b 6

test:$(exe)
		mpiexec -n 1 $(exe) /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_11948.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_11948.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_11948.dat 1 1 128 1
test_6:$(exe)
		jsrun --nrs 1 --tasks_per_rs 6 --cpu_per_rs 6 --gpu_per_rs 6 --rs_per_host 1 $(exe) /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_11948.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_11948.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_11948.dat 6 128 $(p0) $(p1) $b 6
		#mpiexec -n 1 $(exe) /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_11948.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_11948.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_11948.dat 6 128 $(p0) $(p1) $b 6


test_4:$(exe)
		mpiexec -n 4 $(exe) /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_11948.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_11948.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_11948.dat 4 32 $(p0) $(p1) $b 6

test1:$(exe)
	mpiexec -n 1 $(exe) /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_25228.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_25228.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_25228.dat 1 32 $(p0) $(p1) $b 6

test2:$(exe)
	mpiexec -n 1 $(exe) /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_46835.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_46835.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_46835.dat 1 32 $(p0) $(p1) $b 6

test3:$(exe)
	mpiexec -n 1 $(exe) /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_120750.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_120750.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_120750.dat 1 32 $(p0) $(p1) $b 6

test4_4:$(exe)
	mpiexec -n 4 $(exe) /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_659033.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_659033.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_659033.dat 4 32 $(p0) $(p1) $b 6

test4:$(exe)
	mpiexec -n 1 $(exe) /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_659033.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_659033.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_659033.dat 1 10 512 1

test4_6:$(exe)
	jsrun --nrs 1 --tasks_per_rs 6 --cpu_per_rs 6 --gpu_per_rs 6 --rs_per_host 1 $(exe) /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_659033.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_659033.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_659033.dat 6 512 $(p0) $(p1) $b 6
	# mpiexec -n 1 $(exe) /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_659033.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_659033.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_659033.dat 6 512 $(p0) $(p1) $b 6


test4_mem:$(exe)
	nvprof --profile-child-processes mpiexec -n 1 $(exe) /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_659033.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_659033.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_659033.dat 1 32 $(p0) $(p1) $b 6
test5:$(exe)
	mpiexec -n 1 $(exe) /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_1447360.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_1447360.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_1447360.dat 1 32 $(p0) $(p1) $b 6
test6:$(exe)
	mpiexec -n 1 $(exe) /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_21198119.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_21198119.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_21198119.dat 1 32 $(p0) $(p1) $b 6







#	$(mp) -n 2 $(exe)  ../datasets/bcsstk19.mtx_beg_pos.bin ../datasets/bcsstk19.mtx_csr.bin ../datasets/bcsstk19.mtx_weight.bin 
	#		jsrun --nrs $(N_Process) --tasks_per_rs 1 --cpu_per_rs 4 --gpu_per_rs 4 --rs_per_host 1 $(exe) ../colbeg.dat ../colend.dat ../csr.dat $(N_Process)

# bc18_4:$(exe)
# 		jsrun --nrs 4 --tasks_per_rs 1 --cpu_per_rs 1 --gpu_per_rs 1 --rs_per_host 4 $(exe) /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_11948.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_11948.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_11948.dat 4 32 $(p0) $(p1) $b 6 
bc18_4:$(exe)
		jsrun --nrs 1 --tasks_per_rs 4 --cpu_per_rs 4 --gpu_per_rs 4 --rs_per_host 1 $(exe) /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_11948.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_11948.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_11948.dat 4 32 $(p0) $(p1) $b 6

bc18_6:$(exe)
		jsrun --nrs 1 --tasks_per_rs 6 --cpu_per_rs 6 --gpu_per_rs 6 --rs_per_host 1 $(exe) /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_11948.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_11948.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_11948.dat 6 32 $(p0) $(p1) $b 6


bc18_2:$(exe)
		jsrun --nrs 1 --tasks_per_rs 2 --cpu_per_rs 2 --gpu_per_rs 2 --rs_per_host 1 $(exe) /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_11948.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_11948.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_11948.dat 2 32 $(p0) $(p1) $b 6


bc18:$(exe)
		jsrun --nrs 1 --tasks_per_rs 1 --cpu_per_rs 1 --gpu_per_rs 1 --rs_per_host 1 $(exe) /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_11948.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_11948.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_11948.dat 1 32 $(p0) $(p1) $b 6
#
epb2:$(exe)
	jsrun --nrs $(N_Process) --tasks_per_rs 4 --cpu_per_rs 4 --gpu_per_rs 4 --rs_per_host 1 $(exe)  /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_25228.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_25228.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_25228.dat 4 32 6
#
rma10:$(exe)
	jsrun --nrs $(N_Process) --tasks_per_rs 4 --cpu_per_rs 4 --gpu_per_rs 4 --rs_per_host 1 $(exe) /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_46835.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_46835.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_46835.dat 4 32
#

twotone:$(exe)
	jsrun --nrs $(N_Process) --tasks_per_rs 4 --cpu_per_rs 4 --gpu_per_rs 4 --rs_per_host 1 $(exe) ../../../../colbeg_120750.dat ../../../../colend_120750.dat ../../../../csr_120750.dat 4 2000
#
pre2_4:$(exe)
	jsrun --nrs 1 --tasks_per_rs 4 --cpu_per_rs 4 --gpu_per_rs 4 --rs_per_host 1 $(exe)  /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_659033.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_659033.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_659033.dat 4 32 $(p0) $(p1) $b 6 6

pre2:$(exe)
	jsrun --nrs 1 --tasks_per_rs 1 --cpu_per_rs 1 --gpu_per_rs 1 --rs_per_host 1 $(exe)  /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_659033.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_659033.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_659033.dat 1 32 $(p0) $(p1) $b 6 6


pre2_36:$(exe)
	jsrun --nrs 36 --tasks_per_rs 1 --cpu_per_rs 1 --gpu_per_rs 1 --rs_per_host 6 ./mpi_cuda_test.bin /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_659033.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_659033.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_659033.dat 36 32 $(p0) $(p1) $b 6


#pre2_1_:$(exe)
#	jsrun --nrs 4 --tasks_per_rs 1 --cpu_per_rs 1 --gpu_per_rs 1 --rs_per_host 4 $(exe)  /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colbeg_659033.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/colend_659033.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/csr_659033.dat 16 32 1 4 $(p) $(b)

pre2_1:$(exe)
	jsrun --nrs 1 --tasks_per_rs 1 --cpu_per_rs 1 --gpu_per_rs 1 --rs_per_host 1 ./mpi_cuda_test.bin /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/../colbeg_659033.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/../colend_659033.dat /gpfs/alpine/proj-shared/csc289/symbolic/super_dump/superlu_dist-master/EXAMPLE/../csr_659033.dat 1 32 0 0 1



hamre:$(exe)
	jsrun --nrs $(N_Process) --tasks_per_rs 4 --cpu_per_rs 4 --gpu_per_rs 4 --rs_per_host 1 $(exe)  ../../../../colbeg_1447360.dat ../../../../colend_1447360.dat ../../../../csr_1447360.dat 4 2048

huge:$(exe)
	jsrun --nrs $(N_Process) --tasks_per_rs 4 --cpu_per_rs 4 --gpu_per_rs 4 --rs_per_host 1 $(exe)  ../../../../colbeg_21198119.dat ../../../../colend_21198119.dat ../../../../csr_21198119.dat 4 2048

arabic:$(exe)
	jsrun --nrs $(N_Process) --tasks_per_rs 4 --cpu_per_rs 4 --gpu_per_rs 4 --rs_per_host 1 $(exe)  ../../../../colbeg_21198119.dat ../../../../colend_21198119.dat ../../../../csr_21198119.dat 4 2048



#jsrun --nrs $(N_Process) --tasks_per_rs 1 --cpu_per_rs 4 --gpu_per_rs 4 --rs_per_host 1 $(exe) ../datasets/bcsstk19.mtx_beg_pos.bin ../datasets/bcsstk19.mtx_csr.bin ../datasets/bcsstk19.mtx_weight.bin $(N_Process)



#mpirun -n 2 -host localhost $(exe)
clean:
	rm -rf *.o ${exe}


