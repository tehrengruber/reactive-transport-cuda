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

    numeric_t dummy;

    auto reference_result = gibbs_energy(thermo_props, state.n);

    auto optimized_result = gibbs_energy_optimized(thermo_props, state.n);

    std::cout << "H: " << (reference_result.H - optimized_result.H).norm() << std::endl;
    std::cout << "g: " << (reference_result.g - optimized_result.g).norm() << std::endl;

    size_t n_repititions = 10000;

    // run benchmark
    for (size_t i=0; i<n_repititions; ++i) {
        PROFILER_START("gibbs_energy");
        auto result = gibbs_energy(thermo_props, state.n);
        dummy += result.f;
        escape(dummy);
        PROFILER_STOP("gibbs_energy");
    }

    for (size_t i=0; i<n_repititions; ++i) {
        PROFILER_START("gibbs_energy_optimized");
        auto result = gibbs_energy_optimized(thermo_props, state.n);
        dummy += result.f;
        escape(dummy);
        PROFILER_STOP("gibbs_energy_optimized");
    }

    Vector<numeric_t, common::num_species> state_n_copy = state.n;
    for (size_t i=0; i<n_repititions; ++i) {
        clear_cpu_cache();
        gibbs_energy_optimized(thermo_props, state_n_copy);

        PROFILER_START("gibbs_energy_cold_cache");
        auto result = gibbs_energy_optimized(thermo_props, state.n);
        dummy += result.f;
        escape(dummy);
        PROFILER_STOP("gibbs_energy_cold_cache");
    }


    Profiler::IO::print_timings();
}