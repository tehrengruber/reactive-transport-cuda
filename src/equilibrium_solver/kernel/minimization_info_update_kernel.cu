#include "common.h"
#include "common_cuda.h"
#include "equilibrium_solver/minimization_result_info_cuda.h"

namespace equilibrium_solver {

// make variables on the device visible
#ifdef __CUDA_ARCH__
using namespace common_device;
#else
using namespace common;
#endif

constexpr size_t minimization_info_update_batch_size = 128;

__device__ unsigned int min_info_update_counter;

__global__
void minimization_info_update_kernel(MinimizationResultInfoCuda& info, MinizationsOptions& options,
                                     numeric_t* device_Js_ptr,
                                     numeric_t* device_Fs_ptr,
                                     numeric_t* device_Js_ptrs,
                                     numeric_t* device_Fs_ptrs) {
    constexpr size_t bs = minimization_info_update_batch_size;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < (info.n+bs)/bs) {
        // determine problems that have converged and update active_indices array
        for(size_t i=idx*bs; i<idx*(bs+1) && i<info.n; ++i) {
            unsigned int k = atomicInc(min_info_update_counter, 4294967295);
            if (error[i]<options.tol) {
                converged[i] = true;
            } else {
                active_indices[k] = i;
            }
        }

        n_active = k;
    }
}

}