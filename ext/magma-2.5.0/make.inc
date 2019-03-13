#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 2.5.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date January 2019
#//////////////////////////////////////////////////////////////////////////////

# GPU_TARGET contains one or more of Fermi, Kepler, Maxwell, Pascal, Volta
# to specify for which GPUs you want to compile MAGMA:
#     Fermi   - NVIDIA compute capability 2.x cards
#     Kepler  - NVIDIA compute capability 3.x cards
#     Maxwell - NVIDIA compute capability 5.x cards
#     Pascal  - NVIDIA compute capability 6.x cards
#     Volta   - NVIDIA compute capability 7.x cards
# The default is "Kepler Maxwell Pascal".
# Note that NVIDIA no longer supports 1.x cards, as of CUDA 6.5.
# See http://developer.nvidia.com/cuda-gpus
#
#GPU_TARGET ?= Kepler Maxwell Pascal

# --------------------
# programs

CC        = gcc
CXX       = g++
NVCC      = nvcc
FORT      = gfortran

ARCH      = ar
ARCHFLAGS = cr
RANLIB    = ranlib


# --------------------
# flags

# Use -fPIC to make shared (.so) and static (.a) library;
# can be commented out if making only static library.
FPIC      = -fPIC

CFLAGS    = -O3 $(FPIC) -fopenmp -DNDEBUG -DADD_ -Wall -Wno-strict-aliasing -Wshadow -DMAGMA_WITH_MKL
FFLAGS    = -O3 $(FPIC)          -DNDEBUG -DADD_ -Wall -Wno-unused-dummy-argument
F90FLAGS  = -O3 $(FPIC)          -DNDEBUG -DADD_ -Wall -Wno-unused-dummy-argument -x f95-cpp-input
NVCCFLAGS = -O3                  -DNDEBUG -DADD_ -Xcompiler "$(FPIC) -Wall -Wno-unused-function -Wno-strict-aliasing" -std=c++11
LDFLAGS   =     $(FPIC) -fopenmp

# C++11 (gcc >= 4.7) is not required, but has benefits like atomic operations
CXXFLAGS := $(CFLAGS) -std=c++11
CFLAGS   += -std=c99


# --------------------
# libraries

# see MKL Link Advisor at http://software.intel.com/sites/products/mkl/
# gcc/gfortran with MKL 10.3, GNU OpenMP threads (use -fopenmp in CFLAGS, LDFLAGS)
LIB       = -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lstdc++ -lm -lgfortran

# Supposedly, gcc can use Intel threads (libiomp5) instead, but be careful that
# libiomp5 and libgomp are NOT BOTH linked. Above, we use gnu threads as a safer option.
# gcc/gfortran with MKL 10.3, Intel OpenMP threads (remove -fopenmp from LDFLAGS above)
#LIB       = -lmkl_gf_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm -lgfortran

LIB      += -lcublas -lcusparse -lcudart -lcudadevrt


# --------------------
# directories

# define library directories preferably in your environment, or here.
# for MKL run, e.g.: source /opt/intel/composerxe/mkl/bin/mklvars.sh intel64
#MKLROOT ?= /opt/intel/composerxe/mkl
#CUDADIR ?= /usr/local/cuda
-include make.check-mkl
-include make.check-cuda

LIBDIR    = -L$(CUDADIR)/lib64 \
            -L$(MKLROOT)/lib/intel64

INC       = -I$(CUDADIR)/include \
            -I$(MKLROOT)/include