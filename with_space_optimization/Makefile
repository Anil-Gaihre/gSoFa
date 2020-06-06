# exe=./mpi_cuda_test_after_variable_C.bin
exe=./mpi_cuda_test.bin
N_Process=1
#mp=/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/xl-16.1.1-3/spectrum-mpi-10.3.0.0-20190419-q75ow22tialepxlgxcs7esrptqwoheij/bin/mpirun
p0=0
p1=0
b=1
cucc= "$(shell which nvcc)"
cc= "$(shell which mpicxx)"
commflags=-lcudart -L"$(shell dirname $(cucc))"/../lib64
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

test:$(exe)
		mpiexec -n 1 $(exe) ./dataset/colbeg_11948.dat ./dataset/colend_11948.dat ./dataset/csr_11948.dat 1 1
		
pre2:$(exe)
		mpiexec -n 1 $(exe) ../pre2/colbeg_659033.dat ../pre2/colend_659033.dat ../pre2/csr_659033.dat 6 10

#mpirun -n 2 -host localhost $(exe)
clean:
	rm -rf *.o ${exe}


