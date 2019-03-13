cmake_minimum_required (VERSION 2.8.11)
project (ReactiveTransportGPU)

#/usr/include/eigen3

#
# General setup
#
include_directories(3rd_party/matplotlib-cpp /usr/local/lib/python2.7/dist-packages /usr/include/python2.7)
include_directories(include)
include_directories(ext/eigen)
add_compile_options(-DWITHOUT_NUMPY)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register -std=c++14 -march=native")

# Cuda setup
find_package(CUDA QUIET REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_HOST_COMPILER g++-6)
# --generate-line-info -use_fast_math -Xptxas -O3  --generate-line-info  -O3 -maxrregcount=64 -Xptxas -v  -dlcm=cg -maxrregcount=64
set(CUDA_NVCC_FLAGS --expt-relaxed-constexpr --gpu-architecture=compute_61 --gpu-code=sm_61 -G -g -O3)

add_subdirectory("ext/profiler")

#
# Libraries
#
# lib_reactive_transport
add_library(lib_reactive_transport STATIC src/thomas.cpp src/transport.cpp src/chemistry.cpp src/equilibrium_solver/equilibrium_solver.cpp)

# reactive_transport_cuda
#set_source_files_properties( src/chemistry.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )
#set_source_files_properties( src/equilibrium_solver/equilibrium_solver.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )
cuda_add_library(lib_reactive_transport_cuda STATIC src/common_cuda.cu src/chemistry.cu src/equilibrium_solver/equilibrium_solver.cu src/equilibrium_solver/equilibrium_solver_cuda.cu)
target_link_libraries(lib_reactive_transport_cuda lib_reactive_transport)
target_link_libraries(lib_reactive_transport_cuda ${CUDA_LIBRARIES})
target_link_libraries(lib_reactive_transport_cuda cublas curand)

#
# Applications
#
# reactive transport solver - eigen version
add_executable(reactive_transport applications/reactive_transport.cpp)
target_link_libraries(reactive_transport lib_reactive_transport)
target_link_libraries(reactive_transport python2.7)
target_link_libraries(reactive_transport profiler)
target_link_libraries(reactive_transport pcre)

# reactive transport solver refactored version
add_executable(reactive_transport_refactored reactive_transport_refactored.cpp)
target_link_libraries(reactive_transport_refactored lib_reactive_transport)
target_link_libraries(reactive_transport_refactored python2.7)
target_link_libraries(reactive_transport_refactored profiler)
target_link_libraries(reactive_transport_refactored pcre)


# reactive transport solver - cuda version
cuda_add_executable(reactive_transport_cuda reactive_transport.cpp)
target_compile_definitions(reactive_transport_cuda PUBLIC PUBLIC USE_CUDA)
target_link_libraries(reactive_transport_cuda ${CUDA_LIBRARIES})
target_link_libraries(reactive_transport_cuda lib_reactive_transport_cuda)
target_link_libraries(reactive_transport_cuda python2.7)
target_link_libraries(reactive_transport_cuda profiler)
target_link_libraries(reactive_transport_cuda pcre)

#
# Benchmarks
#
add_executable(bench_gibbs_energy benchmarks/gibbs_energy.cpp)
target_link_libraries(bench_gibbs_energy lib_reactive_transport)
target_link_libraries(bench_gibbs_energy python2.7)
target_link_libraries(bench_gibbs_energy profiler)
target_link_libraries(bench_gibbs_energy pcre)

add_executable(boost_fiber_test boost_fiber_test.cpp)
target_link_libraries(boost_fiber_test boost_fiber boost_context pthread)

add_executable(bench_lse_fiber bench_lse_fiber.cpp)
target_link_libraries(bench_lse_fiber boost_fiber boost_context pthread)

cuda_add_executable(cuda_batch_solve cuda_batch_solve.cu)
target_link_libraries(cuda_batch_solve -lcublas -lcurand)
target_link_libraries(cuda_batch_solve profiler)
target_link_libraries(cuda_batch_solve pcre)

cuda_add_executable(cuda_gibbs_energy cuda_gibbs_energy.cu)
target_link_libraries(cuda_gibbs_energy python2.7)
target_link_libraries(cuda_gibbs_energy profiler)
target_link_libraries(cuda_gibbs_energy pcre)

cuda_add_executable(cuda_eigen_lse_test cuda_eigen_lse_test.cu)
target_link_libraries(cuda_batch_solve -lcublas -lcurand)
target_link_libraries(cuda_batch_solve profiler)
target_link_libraries(cuda_batch_solve pcre)

cuda_add_executable(bench_lse_gpu bench_lse_gpu.cu)
target_link_libraries(bench_lse_gpu -lcublas -lcurand)


cuda_add_executable(bench_equilibration benchmarks/equilibration.cpp)
target_compile_definitions(bench_equilibration PUBLIC PUBLIC USE_CUDA)
target_link_libraries(bench_equilibration ${CUDA_LIBRARIES})
target_link_libraries(bench_equilibration lib_reactive_transport_cuda)
target_link_libraries(bench_equilibration python2.7)
target_link_libraries(bench_equilibration profiler)
target_link_libraries(bench_equilibration -lcublas -lcurand)
target_link_libraries(bench_equilibration pcre)

#set(BUILD_SHARED_LIBS 1)
#set(ENV{MKLROOT} "/opt/intel/compilers_and_libraries_2019/linux/mkl/")
#find_package(LAPACKE REQUIRED)
#find_package(MKL REQUIRED)

include_directories(ext)
add_executable(bench_lse bench_lse.cpp)
target_link_libraries(bench_lse profiler)
target_link_libraries(bench_lse pcre)
target_link_libraries(bench_lse pthread)
target_link_libraries(bench_lse ${MKL_LIBRARIES} dl)
#target_include_directories(bench_lse PUBLIC ${MKL_INCLUDE_DIRS})
#target_link_libraries(bench_lse ${LAPACKE_LIBRARIES})
#target_include_directories(bench_lse PUBLIC ${LAPACKE_INCLUDE_DIRS})

add_executable(bench_boost_fiber bench_boost_fiber.cpp)
target_link_libraries(bench_boost_fiber profiler)
target_link_libraries(bench_boost_fiber pcre)
target_link_libraries(bench_boost_fiber boost_fiber boost_context pthread)

add_executable(ringbuf ringbuf.cpp)
