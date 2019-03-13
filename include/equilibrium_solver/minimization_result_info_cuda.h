#ifndef REACTIVETRANSPORTGPU_MINIMIZATION_RESULT_INFO_CUDA
#define REACTIVETRANSPORTGPU_MINIMIZATION_RESULT_INFO_CUDA

#include "common.h"
#include "common_cuda.h"
#include "equilibrium_solver/minimizer_options.h"

namespace equilibrium_solver {

struct MinimizationResultInfoCuda {
    // number of minimization problems
    int n;
    // number of minimization problems that have not converged yet
    int n_active;
    // array of bool containg convergence state of each problem
    bool* converged;
    // the number of iterations of each problem
    int* iterations;
    // number indices of all minimization problems have not converged yet
    int* active_indices;
    // the error of each problem
    float* error;
    MinimizerOptions options;

    bool finished = false;

    MinimizationResultInfoCuda(int n_, MinimizerOptions options_) : n(n_), n_active(n_),
                                                                      options(options_) {
        cudaMallocManaged(&converged, n*sizeof(bool));
        cudaMallocManaged(&iterations, n*sizeof(int));
        cudaMallocManaged(&active_indices, n*sizeof(int));
        cudaMallocManaged(&error, n*sizeof(numeric_t));

        reset();
    }

    void reset() {
        finished = false;
        n_active = n;
        for (int i=0; i<n; ++i) {
            iterations[i] = 0;
            active_indices[i] = i;
            error[i] = std::numeric_limits<numeric_t>::max();
            converged[i] = false;
        }
    }

    ~MinimizationResultInfoCuda() {
        cudaFree(converged);
        cudaFree(iterations);
        cudaFree(active_indices);
        cudaFree(error);
    }

    MinimizationResultInfoCuda(const MinimizationResultInfoCuda&) = delete;
    MinimizationResultInfoCuda& operator=(const MinimizationResultInfoCuda&) = delete;

    void send_to_gpu() {
        int device;
        cudaGetDevice(&device);

        cudaMemPrefetchAsync(this, sizeof(MinimizationResultInfoCuda), device);
        cudaMemPrefetchAsync(converged, n*sizeof(bool), device);
        cudaMemPrefetchAsync(iterations, n*sizeof(int), device);
        cudaMemPrefetchAsync(active_indices, n*sizeof(int), device);
        cudaMemPrefetchAsync(error, n*sizeof(int), device);
        gpuErrchk( cudaPeekAtLastError() );
    }

    void send_to_host() {
        cudaMemPrefetchAsync(this, sizeof(MinimizationResultInfoCuda), cudaCpuDeviceId);
        cudaMemPrefetchAsync(converged, n*sizeof(bool), cudaCpuDeviceId);
        cudaMemPrefetchAsync(iterations, n*sizeof(int), cudaCpuDeviceId);
        cudaMemPrefetchAsync(active_indices, n*sizeof(int), cudaCpuDeviceId);
        cudaMemPrefetchAsync(error, n*sizeof(int), cudaCpuDeviceId);
        gpuErrchk( cudaDeviceSynchronize() );
    }

    void update() {
        // determine problems that have converged and update active_indices array
        int k=0;
        for(int i=0; i<n; ++i) {
            if (!converged[i]) {
                active_indices[k++] = i;
            }
        }
        finished = k==0;
        n_active = k;
    }

    // overload new operator such that instances can be passed by reference to cuda kernels
    void *operator new(size_t len) {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        cudaDeviceSynchronize();
        return ptr;
    }

    // overload delte operator such that instances can be passed by reference to cuda kernels
    void operator delete(void *ptr) {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }
};

}

#endif //REACTIVETRANSPORTGPU_CUDA_BATCH_MINIMIZATION_INFO_H
