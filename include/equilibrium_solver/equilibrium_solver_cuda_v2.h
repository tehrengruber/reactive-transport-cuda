#ifndef REACTIVETRANSPORTGPU_EQUILIBRIUM_SOLVER_CUDA_V2_H
#define REACTIVETRANSPORTGPU_EQUILIBRIUM_SOLVER_CUDA_V2_H

#include <memory>
#include <cublas_v2.h>

#ifdef USE_MAGMA
#include "magma_v2.h"
#endif

#include "common.h"
#include "common_cuda.h"
#include "cuda/simple_cuda_profiler.h"
#include "equilibrium_solver/minimization_result_info_cuda.h"
#include "equilibrium_solver/equilibrium_state.h"
#include "equilibrium_solver/minimization_result_info.h"
#include "equilibrium_solver/equilibrium_solver_cuda.h"
#include "chemistry.h"

namespace equilibrium_solver {

using namespace chemistry;

struct EquilibriumSolverCudaV2 : EquilibriumSolverCudaAbstract {
    using parent_t = EquilibriumSolverCudaAbstract;
    using bs_t = parent_t::bs_t;
    using states_t = parent_t::states_t;

    std::shared_ptr<MinimizationResultInfoCuda> m_info_ptr;

    //
    // device data
    //
    numeric_t* device_Js_ptr;
    numeric_t* device_Fs_ptr;

    ObjectiveResult* device_objective_evals;

    //
    // data used by the lu solver
    //
    cublasHandle_t handle;

    numeric_t** Js_ptrs;
    numeric_t** Fs_ptrs;
    // pointers to the first element of each matrix
    numeric_t** device_Js_ptrs;
    // pointer to the first element of each vector
    numeric_t** device_Fs_ptrs;
    int* device_pivot_arr;
    int* device_info_arr;

    SimpleCudaProfiler profiler;

    #ifdef USE_MAGMA
    magma_queue_t magma_queue;

    magma_int_t* magma_pivot_arr;
    magma_int_t** magma_pivot_arr_ptrs;
    #endif

    EquilibriumSolverCudaV2(size_t ncells_, MinimizerOptions options_) : parent_t(ncells_, options_),
            m_info_ptr(new MinimizationResultInfoCuda(ncells, options)) {
        constexpr size_t t = 2*num_species+num_components;

        // allocate host memory
        Js_ptrs = new numeric_t*[ncells];
        Fs_ptrs = new numeric_t*[ncells];

        // initialize profiler
        profiler.add("transfer component amount H2D");
        profiler.add("transfer states H2D");
        profiler.add("transfer states D2H");
        //profiler.add("gibbs energy kernel");
        profiler.add("assembly");
        profiler.add("minimization update");
        profiler.add("factorization");
        profiler.add("state update");
        profiler.initialize();

        // allocate device memory
        if (cudaMalloc((void**) &device_Js_ptr, ncells*t*t*sizeof(numeric_t)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_Fs_ptr, ncells*t*sizeof(numeric_t)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_Js_ptrs, ncells*sizeof(numeric_t)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_Fs_ptrs, ncells*sizeof(numeric_t)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_pivot_arr, ncells*t*sizeof(int)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_info_arr, ncells*sizeof(int)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_objective_evals, ncells*sizeof(ObjectiveResult)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");

        // init cuBLAS
        cublasStatus_t stat;
        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS initialization failed");
        }

        // init magma
        #ifdef USE_MAGMA
        magma_init();
        int device;
        cudaGetDevice(&device);
        magma_queue_create(device, &magma_queue);

        // allocate pivot arrays
        if (cudaMalloc((void**) &magma_pivot_arr, ncells*t*sizeof(magma_int_t)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");

        if (cudaMalloc((void**) &magma_pivot_arr_ptrs, ncells*sizeof(magma_int_t*)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");

        magma_int_t** pivot_arr_ptrs = new magma_int_t*[ncells];
        for (size_t i=0; i<ncells; ++i) {
            pivot_arr_ptrs[i] = magma_pivot_arr+t*i;
        }

        if (cudaMemcpy((void*) magma_pivot_arr_ptrs, (void*) pivot_arr_ptrs, ncells*sizeof(magma_int_t*), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("Device memory transfer failed");
        }
        delete[] pivot_arr_ptrs;

        #endif
    }

    EquilibriumSolverCudaV2(const EquilibriumSolverCudaV2&) = delete;
    EquilibriumSolverCudaV2& operator=(const EquilibriumSolverCudaV2&) = delete;

    virtual ~EquilibriumSolverCudaV2() {
        profiler.print();

        cublasDestroy(handle);

        cudaFree(device_Fs_ptr);
        cudaFree(device_Fs_ptrs);
        cudaFree(device_Js_ptr);
        cudaFree(device_Js_ptrs);
        cudaFree(device_pivot_arr);
        cudaFree(device_info_arr);
        //cudaFree(device_objective_evals);

        #ifdef USE_MAGMA
        cudaFree(magma_pivot_arr);
        cudaFree(magma_pivot_arr_ptrs);
        magma_queue_destroy(magma_queue);
        magma_finalize();
        #endif

        delete[] Js_ptrs;
        delete[] Fs_ptrs;
    }

    void equilibrate(ThermodynamicProperties thermo_props, bs_t& bs, states_t& states);

    MinimizationResultInfoCuda& info() {
        return *m_info_ptr;
    }
};

}

#endif //REACTIVETRANSPORTGPU_EQUILIBRIUM_SOLVER_CUDA_V2_H
