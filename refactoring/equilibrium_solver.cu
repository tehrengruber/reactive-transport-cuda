#include <memory>
#include <map>
#include <string>
#include <cublas_v2.h>

#include "equilibrium_solver/equilibrium_solver_cuda.h"

#include "simple_cuda_profiler.h"
#include "equilibrium_solver.cpp"
#include "equilibrium_solver/cuda_equilibrium_solver.h"
#include "equilibrium_solver/minimization_result_info_cuda.h"
#include "kernel/equilibrate_cuda_kernel.cu"

#include "simple_timer.h"

namespace equilibrium_solver {

std::vector<MinimizationResultInfo> equilibrate_batch_cuda(ThermodynamicProperties& thermo_props,
                                                           Eigen::Matrix<numeric_t, Eigen::Dynamic, common::num_components, Eigen::RowMajor>& bs,
                                                           EquilibriumStateSOA& states,
                                                           MinimizerOptions options, bool init) {
    assert(init==false);
    size_t ncells = states.size();
    CudaBatchEquilibrationSolver solver(ncells, options);
    auto results = solver.equilibrate(thermo_props, bs, states);

    return results;
}

std::vector<MinimizationResultInfo> equilibrate_batch_cuda_v1(ThermodynamicProperties& thermo_props,
        Eigen::Matrix<numeric_t, Eigen::Dynamic, common::num_components, Eigen::RowMajor>& bs,
        std::vector<EquilibriumState>& states,
        MinimizerOptions options, bool init) {
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t ncells = states.size();
    std::vector<MinimizationResultInfo> results;
    results.resize(ncells);

    numeric_t* device_bs_ptr;
    EquilibriumState* device_states_ptr;
    MinimizationResultInfo* device_results_ptr;

    // allocate device memory
    if (cudaMalloc((void**) &device_bs_ptr, ncells*num_components*sizeof(numeric_t)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
    if (cudaMalloc((void**) &device_states_ptr, ncells*sizeof(EquilibriumState)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
    if (cudaMalloc((void**) &device_results_ptr, ncells*sizeof(MinimizationResultInfo)) != cudaSuccess)
        throw std::runtime_error("Device memory allocation failed");

    // copy data to device
    gpuErrchk( cudaMemcpy((void*) device_bs_ptr, (void*) bs.data(), ncells*num_components*sizeof(numeric_t), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy((void*) device_states_ptr, (void*) states.data(), ncells*sizeof(EquilibriumState), cudaMemcpyHostToDevice) );

    // compute block size and grid size
    int blockSize;
    int minGridSize;
    int gridSize;
    cudaOccupancyMaxPotentialBlockSize(
            &minGridSize,
            &blockSize,
            (void*)equilibrate_cuda_kernel,
            0,
            ncells);
    gridSize = (ncells + blockSize - 1) / blockSize;

    {
        int numBlocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks,
            (void*) equilibrate_cuda_kernel,
            blockSize,
            0);

        int device;
        cudaDeviceProp prop;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);

        std::cout << "Occupancy:" << double(numBlocks * blockSize)/prop.maxThreadsPerMultiProcessor << std::endl;
    }

    // call actual kernel
    cudaEventRecord(start);
    equilibrate_cuda_kernel<<<gridSize, blockSize>>>(thermo_props, ncells, device_bs_ptr, device_states_ptr, options, init, device_results_ptr);
    cudaEventRecord(stop);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // copy data back to host
    gpuErrchk(cudaMemcpy((void*) states.data(), (void*) device_states_ptr, ncells*sizeof(EquilibriumState), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy((void*) results.data(), (void*) device_results_ptr, ncells*sizeof(MinimizationResultInfo), cudaMemcpyDeviceToHost));

    // free memory
    if (cudaFree((void*) device_bs_ptr) != cudaSuccess)
            throw std::runtime_error("Device memory deallocation failed");
    if (cudaFree((void*) device_states_ptr) != cudaSuccess)
            throw std::runtime_error("Device memory deallocation failed");
    if (cudaFree((void*) device_results_ptr) != cudaSuccess)
        throw std::runtime_error("Device memory deallocation failed");

    // output throughput in processed cells/second
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    /*std::cout << "processed cells/s: " << 1000*ncells/milliseconds << std::endl;

    size_t total_iterations=0;
    for (size_t i=0; i<ncells; ++i) {
        total_iterations += results[i].it;
    }
    std::cout << "iterations/s: " << 1000*total_iterations/milliseconds << std::endl;*/

    return results;
}

}