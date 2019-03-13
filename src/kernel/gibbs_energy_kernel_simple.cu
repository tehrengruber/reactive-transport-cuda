#include "common.h"
#include "common_cuda.h"
#include "chemistry.h"
#include "equilibrium_solver/minimizer_options.h"
#include "equilibrium_solver/equilibrium_state.h"

namespace chemistry {

// make variables on the device visible
#ifdef __CUDA_ARCH__
    using namespace common_device;
#else
    using namespace common;
#endif

__global__
void gibbs_energy_kernel_simple(ThermodynamicProperties thermo_props,
                         int nevals,
                         Vector<numeric_t, common::num_species>* xs,
                         ObjectiveResult* results) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;

    if (8*tidx < nevals) {
        for (size_t i=0; i<8; ++i) {
            int idx = 8*tidx+i;

            gibbs_energy_opt_inplace_gradient(thermo_props, xs[idx], results[idx].g);
            gibbs_energy_opt_inplace_hessian(thermo_props, xs[idx], results[idx].H);
        }
    }
}

}