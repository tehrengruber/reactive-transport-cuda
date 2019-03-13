#include "common.h"
#include "common_cuda.h"
#include "chemistry.h"
#include "equilibrium_solver/minimizer_options.h"
#include "equilibrium_solver/equilibrium_state.h"
#include "equilibrium_solver/equilibrium_solver.h"

namespace equilibrium_solver {

// make variables on the device visible
#ifdef __CUDA_ARCH__
    using namespace common_device;
#else
    using namespace common;
#endif

using chemistry::ThermodynamicProperties;

__global__
void equilibrate_cuda_kernel(ThermodynamicProperties thermo_props,
                             size_t ncells,
                             numeric_t* bs_raw,
                             EquilibriumState* states,
                             MinimizerOptions options,
                             bool init,
                             MinimizationResultInfo* results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < ncells) {
        Eigen::Map<component_amounts_t> b(bs_raw+num_components*idx);
        results[idx] = equilibrate(thermo_props, b, *(states+idx), options, init);
    }
}

}