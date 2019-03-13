#include "common.h"
#include "common_cuda.h"
#include "chemistry.h"
#include "equilibrium_solver/minimizer_options.h"
#include "equilibrium_solver/equilibrium_state.h"

namespace equilibrium_solver {

// make variables on the device visible
#ifdef __CUDA_ARCH__
using namespace common_device;
#else
using namespace common;
#endif

using chemistry::ThermodynamicProperties;

__global__
void gibbs_energy_kernel(ThermodynamicProperties thermo_props, size_t ncells,
                         size_t* active_ncell_indices, EquilibriumState* states, ObjectiveResult* results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < ncells) {
        int cidx = active_ncell_indices[idx];
        EquilibriumState& state = states[cidx];

        results[cidx] = gibbs_energy_optimized(thermo_props, state.x);
    }
}

}