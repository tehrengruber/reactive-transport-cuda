#include "profiler/profiler.hpp"
#include "profiler/profiler_io.cpp"

#include "chemistry.h"
#include "equilibrium_solver/equilibrium_solver.h"
#include "equilibrium_solver/minimizer_options.h"
#include "equilibrium_solver/equilibrium_state.h"
#include "misc/simple_timer.h"

using chemistry::component_amounts;
using chemistry::gibbs_energy;
using chemistry::gibbs_energy_optimized;
using chemistry::ThermodynamicProperties;
using equilibrium_solver::MinimizerOptions;
using equilibrium_solver::EquilibriumState;
using equilibrium_solver::equilibrate;

int main() {
    common_device::initialize();

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

    size_t nevals = 1000000;

    Vector<numeric_t, common::num_species>*    device_states_x; // alias to amounts of species in mol
    Vector<numeric_t, common::num_species>*    states_x = new Vector<numeric_t, common::num_species>[nevals];
    for (size_t i=0; i<nevals; ++i) {
        states_x[i] = state.x;
    }
    chemistry::ObjectiveResult* device_obj_results;
    chemistry::ObjectiveResult* obj_results = new chemistry::ObjectiveResult[nevals];

    if (cudaMalloc((void**) &device_states_x, nevals*sizeof(Vector<numeric_t, common::num_species>)) != cudaSuccess)
        throw std::runtime_error("Device memory allocation failed");
    if (cudaMalloc((void**) &device_obj_results, nevals*sizeof(chemistry::ObjectiveResult)) != cudaSuccess)
        throw std::runtime_error("Device memory allocation failed");
    if (cudaMemcpy((void*) device_states_x, (void*) states_x, nevals*sizeof(Vector<numeric_t, common::num_species>), cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("Device memory transfer");
    }

    std::cout << "nevals: " << nevals << std::endl;

    // bench gpu
    SimpleTimer t;
    t.tic();
    gibbs_energy_batch_gpu(thermo_props, nevals, device_states_x, device_obj_results);
    t.toc();
    std::cout << "time gpu: " << t.duration() << std::endl;
    std::cout << "throughput gpu: " << nevals/t.duration() << std::endl;

    // bench cpu
    t.tic();
    for (size_t i=0; i<nevals; ++i) {
        gibbs_energy_opt_inplace_gradient(thermo_props, states_x[0], obj_results[0].g);
        gibbs_energy_opt_inplace_hessian(thermo_props, states_x[0], obj_results[0].H);
        escape(obj_results[i]);
    }
    t.toc();
    std::cout << "time cpu: " << t.duration() << std::endl;
    std::cout << "throughput cpu: " << nevals/t.duration() << std::endl;
}