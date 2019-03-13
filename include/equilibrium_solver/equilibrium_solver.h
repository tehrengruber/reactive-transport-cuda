#ifndef REACTIVETRANSPORTGPU_EQUILIBRIUM_SOLVER_H
#define REACTIVETRANSPORTGPU_EQUILIBRIUM_SOLVER_H

#include "common.h"
#include "common_cuda.h"
#include "chemistry.h"
#include "equilibrium_solver/minimizer.h"
#include "equilibrium_solver/equilibrium_state.h"
#include "equilibrium_solver/equilibrium_solver.h"
#include "equilibrium_solver/minimizer_options.h"
#include "equilibrium_solver/minimization_result_info.h"

#ifdef USE_MULTIPLE_THREADS
#include <omp.h>
#endif

namespace equilibrium_solver {

// make variables on the device visible
#ifdef __CUDA_ARCH__
using namespace common_device;
#else
using namespace common;
#endif

using chemistry::ThermodynamicProperties;
using chemistry::gibbs_energy;
using chemistry::gibbs_energy_optimized;
using chemistry::gibbs_energy_pure_phases;

// Calculate the equilibrium state of the chemical system.
// Parameters:
//   - T is temperature in K
//   - P is pressure in Pa
//   - b is an array with molar amounts of components
//   - options is a dictionary with options for the calculation
// Return:
//   - an state object with members n, y, z, so that the
//     molar amounts of the species is given by state.n,
//     and the Lagrange multipliers by state.y and state.z.
template<typename EQUI_STATE_T>
DEVICE_DECL_SPEC HOST_DECL_SPEC
MinimizationResultInfo equilibrate(const ThermodynamicProperties &thermo_props,
                                   component_amounts_t &b,
                                   EQUI_STATE_T &&state,
                                   const MinimizerOptions &options,
                                   bool init = true) {
    ST_VER_ONLY(HOST_VER_ONLY(PROFILER_START("equilibrate")));
    const numeric_t T = thermo_props.T;
    const numeric_t P = thermo_props.P;

    // Check if state is None, and if so, find an initial guess
    if (init) {
        // Define the objective function to calculate an initial guess
        auto objective = [&](auto x) {
            return gibbs_energy_pure_phases(thermo_props, x);
        };

        // Define the minimization problem
        OptimumProblem<decltype(objective)> problem(objective, formula_matrix, b);

        // Minimize the Gibbs energy assuming each species is a pure phase
        minimize(state, problem, options, init);
        /*state.y *= R * T;
        state.z *= R * T;*/
    }

    /*state.y /= R * T;
    state.z /= R * T;*/

    // Define now the objective function with a normalized Gibbs energy function
    auto objective = [&](auto x) {
        return gibbs_energy_optimized(thermo_props, x);
    };

    // Define the minimization problem
    OptimumProblem<decltype(objective)> problem(objective, formula_matrix, b);

    // Minimize the Gibbs energy of the chemical system
    auto result = minimize(state, problem, options);

    // Finalize the setting of the equilibrium state
    ST_VER_ONLY(HOST_VER_ONLY(PROFILER_START("equilibrate.state_update")));
    /*state.n = state.x;
    state.m = chemistry::masses(state.x);
    state.y = state.y * R * T;
    state.z = state.z * R * T;
    state.a = chemistry::ln_activities(T, P, state.x).array().exp().matrix();
    state.c = chemistry::concentrations(T, P, state.x);
    state.g = (state.a.array() / state.c.array()).matrix();*/
    ST_VER_ONLY(HOST_VER_ONLY(PROFILER_STOP("equilibrate.state_update")));

    ST_VER_ONLY(HOST_VER_ONLY(PROFILER_STOP("equilibrate")));
    return result;
}

}

#endif //REACTIVETRANSPORTGPU_EQUILIBRIUM_SOLVER_H
