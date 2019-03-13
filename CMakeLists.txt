cmake_minimum_required (VERSION 2.8.11)
project (ReactiveTransportGPU)

#/usr/include/eigen3

#
# General setup
#
include_directories(3rd_party/matplotlib-cpp /usr/local/lib/python2.7/dist-packages /usr/include/python2.7)
include_directories(/cluster/apps/python/2.7.14/x86_64/include/python2.7)
link_directories(/cluster/apps/python/2.7.14/x86_64/lib64/)

include_directories(include)
include_directories(ext/eigen)
include_directories(ext)
add_compile_options(-DWITHOUT_NUMPY)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register -std=c++14 -march=native -fopenmp")

# Cuda setup
if(${USE_CUDA})
    set(CUDA_BIN_PATH /usr/local/cuda)
    find_package(CUDA QUIET REQUIRED)
    set(CUDA_SEPARABLE_COMPILATION ON)
    set(CUDA_HOST_COMPILER g++-6)
    # --generate-line-info -use_fast_math -Xptxas -O3  --generate-line-info  -O3 -maxrregcount=64 -Xptxas -v  -dlcm=cg -maxrregcount=64  -Xptxas -O3
    # --gpu-architecture=compute_61 --gpu-code=sm_61 -O3 -Xptxas -O3
    set(CUDA_NVCC_FLAGS -DWITHOUT_NUMPY --expt-relaxed-constexpr)

    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture=compute_61 --gpu-code=sm_61)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture=compute_70 --gpu-code=sm_70)
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture=compute_37 --gpu-code=sm_37)

    if(${CMAKE_BUILD_TYPE} MATCHES DEBUG)
        message("DEBUG build")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G)
    else()
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -Xptxas -O3)
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -DADD_ -Wno-strict-aliasing")
    include_directories(/usr/local/cuda/include /opt/intel/compilers_and_libraries_2019.2.187/linux/mkl/include /usr/local/magma/include)
    link_directories(/usr/local/cuda/lib64 /opt/intel/compilers_and_libraries_2019.2.187/linux/mkl/lib/intel64 /usr/local/magma/lib)
    set(MAGMA_LIBS mkl_gf_lp64 mkl_gnu_thread mkl_core pthread stdc++ m gfortran cublas cusparse cudart cudadevrt magma)
    SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
    SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
endif()

add_subdirectory("ext/profiler")


#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -I/usr/local/magma/include -DNDEBUG -DADD_ -I/usr/local/cuda/include
#        -I/opt/intel/compilers_and_libraries_2019.2.187/linux/mkl/include
#        -L/usr/local/magma/lib -lmagma_sparse -lmagma -L/usr/local/cuda/lib64
#        -L/opt/intel/compilers_and_libraries_2019.2.187/linux/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread
#        -lmkl_core -lsvml -lm -lcublas -lcusparse -lcudart -lcudadevrt
#        )
#
#set(CMAKE_EXE_LINKER_FLAGS
#        "-L/usr/local/magma/lib -lmagma_sparse -lmagma -L/usr/local/cuda/lib64 -L/opt/intel/compilers_and_libraries_2019.2.187/linux/compiler/lib/intel64 -L/opt/intel/compilers_and_libraries_2019.2.187/linux/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lsvml -lm -lcublas -lcusparse -lcudart -lcudadevrt"
#        )

# Magma setup
#find_package(MAGMA QUIET REQUIRED)
#include_directories(${MAGMA_INCLUDE_DIR})

#
# Libraries
#
# lib_reactive_transport
add_library(lib_reactive_transport STATIC src/thomas.cpp src/transport.cpp src/chemistry.cpp)

if(${USE_CUDA})
    # reactive_transport_cuda
    #set_source_files_properties( src/chemistry.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )
    #set_source_files_properties( src/equilibrium_solver/equilibrium_solver.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )
    cuda_add_library(lib_reactive_transport_cuda_v1 STATIC src/common_cuda.cu
            src/chemistry.cu
            src/equilibrium_solver/equilibrium_solver_cuda_v1.cu)
    target_link_libraries(lib_reactive_transport_cuda_v1 lib_reactive_transport)
    target_link_libraries(lib_reactive_transport_cuda_v1 ${CUDA_LIBRARIES})
    target_link_libraries(lib_reactive_transport_cuda_v1 cublas curand)
    target_compile_definitions(lib_reactive_transport_cuda_v1 PUBLIC USE_CUDA USE_CUDA_IMPL_1)

    cuda_add_library(lib_reactive_transport_cuda_v2 STATIC src/common_cuda.cu
            src/chemistry.cu
            src/equilibrium_solver/equilibrium_solver_cuda_v2.cu)
    target_link_libraries(lib_reactive_transport_cuda_v2 lib_reactive_transport)
    target_link_libraries(lib_reactive_transport_cuda_v2 ${CUDA_LIBRARIES})
    target_link_libraries(lib_reactive_transport_cuda_v2 cublas curand)
    target_compile_definitions(lib_reactive_transport_cuda_v2 PUBLIC USE_CUDA USE_CUDA_IMPL_2)

    cuda_add_library(lib_reactive_transport_cuda_magma_v2 STATIC src/common_cuda.cu
            src/chemistry.cu
            src/equilibrium_solver/equilibrium_solver_cuda_v2.cu)
    target_link_libraries(lib_reactive_transport_cuda_magma_v2 lib_reactive_transport)
    target_link_libraries(lib_reactive_transport_cuda_magma_v2 ${CUDA_LIBRARIES})
    target_link_libraries(lib_reactive_transport_cuda_magma_v2 cublas curand)
    target_compile_definitions(lib_reactive_transport_cuda_magma_v2 PUBLIC USE_CUDA USE_CUDA_IMPL_2 USE_MAGMA USE_SHARED_MEM)

    cuda_add_library(lib_reactive_transport_cuda_sm_v2 STATIC src/common_cuda.cu
            src/chemistry.cu
            src/equilibrium_solver/equilibrium_solver_cuda_v2.cu)
    target_link_libraries(lib_reactive_transport_cuda_sm_v2 lib_reactive_transport)
    target_link_libraries(lib_reactive_transport_cuda_sm_v2 ${CUDA_LIBRARIES})
    target_link_libraries(lib_reactive_transport_cuda_sm_v2 cublas curand)
    target_compile_definitions(lib_reactive_transport_cuda_sm_v2 PUBLIC USE_SHARED_MEM USE_CUDA USE_CUDA_IMPL_2)
endif()

#
# Applications
#
# reactive transport solver - eigen version
add_executable(reactive_transport applications/reactive_transport.cpp)
target_link_libraries(reactive_transport lib_reactive_transport)
target_link_libraries(reactive_transport python2.7)
target_link_libraries(reactive_transport profiler)
target_link_libraries(reactive_transport pcre)


add_executable(reactive_transport_mt applications/reactive_transport.cpp)
target_compile_definitions(reactive_transport_mt PUBLIC USE_MULTIPLE_THREADS)
target_compile_options(reactive_transport_mt PUBLIC -fopenmp)
target_link_libraries(reactive_transport_mt lib_reactive_transport)
target_link_libraries(reactive_transport_mt python2.7)
target_link_libraries(reactive_transport_mt profiler)
target_link_libraries(reactive_transport_mt pcre)

add_executable(reactive_transport_runtime_profile applications/reactive_transport_runtime_profile.cpp)
target_link_libraries(reactive_transport_runtime_profile lib_reactive_transport)
target_link_libraries(reactive_transport_runtime_profile python2.7)
target_link_libraries(reactive_transport_runtime_profile profiler)
target_link_libraries(reactive_transport_runtime_profile pcre)

if(${USE_CUDA})
    # reactive transport solver - cuda version
    cuda_add_executable(reactive_transport_cuda applications/reactive_transport_cuda.cu)
    target_compile_definitions(reactive_transport_cuda PUBLIC USE_CUDA USE_CUDA_IMPL_1)
    target_link_libraries(reactive_transport_cuda ${CUDA_LIBRARIES})
    target_link_libraries(reactive_transport_cuda lib_reactive_transport_cuda_v1)
    target_link_libraries(reactive_transport_cuda python2.7)
    target_link_libraries(reactive_transport_cuda profiler)
    target_link_libraries(reactive_transport_cuda pcre)

    # reactive transport solver - cuda version
    cuda_add_executable(reactive_transport_cuda_v2 applications/reactive_transport_cuda.cu)
    target_compile_definitions(reactive_transport_cuda_v2 PUBLIC USE_CUDA USE_CUDA_IMPL_2)
    target_link_libraries(reactive_transport_cuda_v2 ${CUDA_LIBRARIES})
    target_link_libraries(reactive_transport_cuda_v2 lib_reactive_transport_cuda_v2)
    target_link_libraries(reactive_transport_cuda_v2 python2.7)
    target_link_libraries(reactive_transport_cuda_v2 profiler)
    target_link_libraries(reactive_transport_cuda_v2 pcre)

    cuda_add_executable(reactive_transport_cuda_sm_v2 applications/reactive_transport_cuda.cu)
    target_compile_definitions(reactive_transport_cuda_sm_v2 PUBLIC USE_CUDA USE_SHARED_MEM USE_CUDA_IMPL_2)
    target_link_libraries(reactive_transport_cuda_sm_v2 ${CUDA_LIBRARIES})
    target_link_libraries(reactive_transport_cuda_sm_v2 lib_reactive_transport_cuda_sm_v2)
    target_link_libraries(reactive_transport_cuda_sm_v2 python2.7)
    target_link_libraries(reactive_transport_cuda_sm_v2 profiler)
    target_link_libraries(reactive_transport_cuda_sm_v2 pcre)

    cuda_add_executable(reactive_transport_cuda_magma_v2 applications/reactive_transport_cuda.cu)
    target_compile_definitions(reactive_transport_cuda_magma_v2 PUBLIC USE_MAGMA USE_SHARED_MEM USE_CUDA USE_CUDA_IMPL_2)
    target_link_libraries(reactive_transport_cuda_magma_v2 ${CUDA_LIBRARIES})
    target_link_libraries(reactive_transport_cuda_magma_v2 lib_reactive_transport_cuda_magma_v2)
    target_link_libraries(reactive_transport_cuda_magma_v2 python2.7)
    target_link_libraries(reactive_transport_cuda_magma_v2 profiler)
    target_link_libraries(reactive_transport_cuda_magma_v2 pcre)
    target_link_libraries(reactive_transport_cuda_magma_v2 ${MAGMA_LIBS})
endif()

#
# Benchmarks
#
add_executable(bench_gibbs_energy benchmarks/bench_gibbs_energy.cpp)
target_link_libraries(bench_gibbs_energy lib_reactive_transport)
target_link_libraries(bench_gibbs_energy python2.7)
target_link_libraries(bench_gibbs_energy profiler)
target_link_libraries(bench_gibbs_energy pcre)

add_executable(bench_reactive_transport_cpu benchmarks/bench_reactive_transport_cpu.cpp)
target_link_libraries(bench_reactive_transport_cpu lib_reactive_transport)
target_link_libraries(bench_reactive_transport_cpu python2.7)
target_link_libraries(bench_reactive_transport_cpu profiler)
target_link_libraries(bench_reactive_transport_cpu pcre)

add_executable(bench_boost_fiber benchmarks/bench_boost_fiber.cpp)
target_link_libraries(bench_boost_fiber boost_fiber boost_context)
target_link_libraries(bench_boost_fiber profiler)
target_link_libraries(bench_boost_fiber pcre)

add_executable(bench_smf benchmarks/bench_smf.cpp)
target_link_libraries(bench_smf profiler)
target_link_libraries(bench_smf pcre)

add_executable(bench_reactive_transport_cpu_mt benchmarks/bench_reactive_transport_cpu.cpp)
target_compile_definitions(bench_reactive_transport_cpu_mt PUBLIC USE_MULTIPLE_THREADS)
target_compile_options(bench_reactive_transport_cpu_mt PUBLIC -fopenmp)
target_link_libraries(bench_reactive_transport_cpu_mt lib_reactive_transport)
target_link_libraries(bench_reactive_transport_cpu_mt python2.7)
target_link_libraries(bench_reactive_transport_cpu_mt profiler)
target_link_libraries(bench_reactive_transport_cpu_mt pcre)

if(${USE_CUDA})
    # reactive transport solver - cuda version
    cuda_add_executable(bench_reactive_transport_cuda_v1 benchmarks/bench_reactive_transport_cuda.cu)
    target_compile_definitions(bench_reactive_transport_cuda_v1 PUBLIC USE_CUDA USE_CUDA_IMPL_1)
    target_link_libraries(bench_reactive_transport_cuda_v1 ${CUDA_LIBRARIES})
    target_link_libraries(bench_reactive_transport_cuda_v1 lib_reactive_transport_cuda_v1)
    target_link_libraries(bench_reactive_transport_cuda_v1 python2.7)
    target_link_libraries(bench_reactive_transport_cuda_v1 profiler)
    target_link_libraries(bench_reactive_transport_cuda_v1 pcre)

    cuda_add_executable(bench_reactive_transport_cuda_v2 benchmarks/bench_reactive_transport_cuda.cu)
    target_compile_definitions(bench_reactive_transport_cuda_v2 PUBLIC USE_CUDA USE_CUDA_IMPL_2)
    target_link_libraries(bench_reactive_transport_cuda_v2 ${CUDA_LIBRARIES})
    target_link_libraries(bench_reactive_transport_cuda_v2 lib_reactive_transport_cuda_v2)
    target_link_libraries(bench_reactive_transport_cuda_v2 python2.7)
    target_link_libraries(bench_reactive_transport_cuda_v2 profiler)
    target_link_libraries(bench_reactive_transport_cuda_v2 pcre)

    cuda_add_executable(bench_reactive_transport_cuda_sm_v2 benchmarks/bench_reactive_transport_cuda.cu)
    target_compile_definitions(bench_reactive_transport_cuda_sm_v2 PUBLIC USE_CUDA USE_SHARED_MEM USE_CUDA_IMPL_2)
    target_link_libraries(bench_reactive_transport_cuda_sm_v2 ${CUDA_LIBRARIES})
    target_link_libraries(bench_reactive_transport_cuda_sm_v2 lib_reactive_transport_cuda_sm_v2)
    target_link_libraries(bench_reactive_transport_cuda_sm_v2 python2.7)
    target_link_libraries(bench_reactive_transport_cuda_sm_v2 profiler)
    target_link_libraries(bench_reactive_transport_cuda_sm_v2 pcre)

    cuda_add_executable(bench_reactive_transport_cuda_magma_v2 benchmarks/bench_reactive_transport_cuda.cu)
    target_compile_definitions(bench_reactive_transport_cuda_magma_v2 PUBLIC USE_CUDA USE_CUDA_IMPL_2 USE_MAGMA USE_SHARED_MEM)
    target_link_libraries(bench_reactive_transport_cuda_magma_v2 ${CUDA_LIBRARIES})
    target_link_libraries(bench_reactive_transport_cuda_magma_v2 lib_reactive_transport_cuda_magma_v2)
    target_link_libraries(bench_reactive_transport_cuda_magma_v2 python2.7)
    target_link_libraries(bench_reactive_transport_cuda_magma_v2 profiler)
    target_link_libraries(bench_reactive_transport_cuda_magma_v2 pcre)
    target_link_libraries(bench_reactive_transport_cuda_magma_v2 ${MAGMA_LIBS})

    cuda_add_executable(bench_lse_gpu benchmarks/bench_lse_gpu.cu)
    target_link_libraries(bench_lse_gpu ${CUDA_LIBRARIES})
    target_link_libraries(bench_lse_gpu -lcublas -lcurand)
    target_link_libraries(bench_lse_gpu profiler)
    target_link_libraries(bench_lse_gpu pcre)
    target_link_libraries(bench_lse_gpu ${MAGMA_LIBS})

    cuda_add_executable(bench_gibbs_energy_gpu benchmarks/bench_gibbs_energy_gpu.cu)
    target_link_libraries(bench_gibbs_energy_gpu ${CUDA_LIBRARIES})
    target_link_libraries(bench_gibbs_energy_gpu lib_reactive_transport_cuda_v1)
    target_link_libraries(bench_gibbs_energy_gpu profiler)
    target_link_libraries(bench_gibbs_energy_gpu pcre)
endif()