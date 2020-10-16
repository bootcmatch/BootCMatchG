# CUB
#CUB_PATH = ${HOME}/cub-1.7.4
CUB_PATH = 
# NSPARSE
NSPARSE_PATH = EXTERNAL/nsparse-master/cuda-c
NSPARSE_GPU_ARCH = -arch=sm_60
GPU_ARCH = -arch=sm_60
#-------------------------------------------------
CC = nvcc
NVCC_FLAG = --compiler-options -Wall
OBJECTS = $(BUILDDIR)/utils.o $(BUILDDIR)/nsparse.o  $(BUILDDIR)/vector.o $(BUILDDIR)/CSR.o $(BUILDDIR)/matrixIO.o
LIBS = -lcusparse -lcublas -lcusolver -lcurand
INCLUDE = -I. -I$(NSPARSE_PATH)/inc -I$(CUB_PATH)

# Bilding dirs
BUILDDIR    := obj
TARGETDIR   := bin


$(TARGETDIR)/solve_example: solve_example.cu $(OBJECTS) $(OBJECTS_AFSAI)
ifndef CUB_PATH
$(error CUB PATH must be set)
endif
	mkdir -p $(TARGETDIR) 
	$(CC) -o $@  $(INCLUDE) $(LIBS) $(GPU_ARCH) $(NVCC_FLAG) $^

$(BUILDDIR)/vector.o: matrix/vector.cu
	mkdir -p $(BUILDDIR) 
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^

$(BUILDDIR)/matrixIO.o: matrix/matrixIO.cu
	mkdir -p $(BUILDDIR) 
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^

$(BUILDDIR)/CSR.o: matrix/CSR.cu
	mkdir -p $(BUILDDIR) 
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^

$(BUILDDIR)/utils.o: utility/utils.cu
	mkdir -p $(BUILDDIR) 
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^

$(BUILDDIR)/nsparse.o: $(NSPARSE_PATH)/src/kernel/kernel_spgemm_hash_d.cu
	mkdir -p $(BUILDDIR) 
	$(CC) -c -o $@ $(LIBS) $(INCLUDE) $(NSPARSE_GPU_ARCH) $^

clean:
	rm $(BUILDDIR)/*.o $(TARGETDIR)/*
