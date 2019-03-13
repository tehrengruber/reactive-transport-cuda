#include "profiler/profiler.hpp"
#include "profiler/profiler_io.cpp"

#include "chemistry.h"
#include "equilibrium_solver/equilibrium_solver.h"
#include "equilibrium_solver/minimizer_options.h"
#include "equilibrium_solver/equilibrium_state.h"

using chemistry::component_amounts;
using chemistry::gibbs_energy;
using chemistry::ThermodynamicProperties;
using equilibrium_solver::MinimizerOptions;
using equilibrium_solver::EquilibriumState;
using equilibrium_solver::equilibrate;

int main() {
    // compute the amount of species for a simple setup
    MinimizerOptions options;
    EquilibriumState state;
    ThermodynamicProperties thermo_props(60.0 + 273.15, 100 * 1e5);
    auto b = component_amounts(1.0, 0.05, 0.0, 0.0);
    equilibrate(thermo_props, b, state, options, true);

    size_t n_repititions = 10000;

    for (size_t i=0; i<n_repititions; ++i) {
        equilibrate(thermo_props, b, state, options, true);
    }
}