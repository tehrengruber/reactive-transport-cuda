#include "common.h"
#include "chemistry.h"
#include "equilibrium_solver/minimizer.h"
#include "equilibrium_solver/equilibrium_state.h"
#include "equilibrium_solver/equilibrium_solver.h"
#include "equilibrium_solver/minimizer_options.h"
#include "equilibrium_solver/minimization_result_info.h"

namespace equilibrium_solver {

using chemistry::ThermodynamicProperties;
using chemistry::gibbs_energy;
using chemistry::gibbs_energy_optimized;
using chemistry::gibbs_energy_pure_phases;

#ifndef __CUDA_ARCH__
std::vector<MinimizationResultInfo> equilibrate_batch(const ThermodynamicProperties& thermo_props,
        Eigen::Matrix<numeric_t, Eigen::Dynamic, common::num_components, Eigen::RowMajor>& bs,
        std::vector<EquilibriumState>& states,
        MinimizerOptions options, bool init) {
    // initialize variables
    size_t ncells = states.size();
    std::vector<MinimizationResultInfo> results;
    results.reserve(ncells);

    // equilibrate
    for (size_t icell=0; icell<ncells; ++icell) {
        component_amounts_t b = bs.row(icell);
        results.push_back(equilibrate(thermo_props, b, states[icell], options, init));
    }

    return results;
}

std::vector<MinimizationResultInfo> equilibrate_batch(const ThermodynamicProperties& thermo_props,
                                                      Eigen::Matrix<numeric_t, Eigen::Dynamic, common::num_components, Eigen::RowMajor>& bs,
                                                      EquilibriumStateSOA& states,
                                                      MinimizerOptions options, bool init) {
    // initialize variables
    size_t ncells = states.size();
    std::vector<MinimizationResultInfo> results;
    results.reserve(ncells);

    // equilibrate
    for (size_t icell=0; icell<ncells; ++icell) {
        component_amounts_t b = bs.row(icell);
        results.push_back(equilibrate(thermo_props, b, states[icell], options, init));
    }

    return results;
}
#endif

}