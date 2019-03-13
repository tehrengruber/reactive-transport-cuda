#include "common.h"
#include "common_cuda.h"
#include "chemistry.h"
#include "equilibrium_solver/minimizer_options.h"
#include "equilibrium_solver/equilibrium_state.h"
#include "equilibrium_solver/minimization_result_info_cuda.h"

namespace equilibrium_solver {

// make variables on the device visible
#ifdef __CUDA_ARCH__
    using namespace common_device;
#else
    using namespace common;
#endif

__global__
void state_update_kernel(ThermodynamicProperties thermo_props,
                         MinimizerOptions options,
                         MinimizationResultInfoCuda& info,
                         Vector<numeric_t, common::num_species>* xs,
                         Vector<numeric_t, common::num_components>* ys,
                         Vector<numeric_t, common::num_species>* zs,
                         numeric_t* deltas) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < info.n_active) {
        int cidx = info.active_indices[idx];
        // initialize variables
        Map<Vector<numeric_t, 2*num_species+num_components>> delta(deltas+cidx*(2*num_species+num_components));

        auto& x = xs[cidx];
        auto& y = ys[cidx];
        auto& z = zs[cidx];

        const numeric_t tau = options.tau;

        Eigen::Ref<Vector<numeric_t, num_species>> dx = delta.head(num_species);
        Eigen::Ref<Vector<numeric_t, num_components>> dy = delta.segment(num_species, num_components);
        Eigen::Ref<Vector<numeric_t, num_species>> dz = delta.tail(num_species);

        // Calculate the new values for x and z
        for (size_t i=0; i<num_species; ++i) {
            x[i] += (x[i] + dx[i] > 0.0) ? dx[i] : (-tau * x[i]);
            z[i] += (z[i] + dz[i] > 0.0) ? dz[i] : (-tau * z[i]);
        }

        // Calculate the new values for y
        y += dy;
    }
}

}