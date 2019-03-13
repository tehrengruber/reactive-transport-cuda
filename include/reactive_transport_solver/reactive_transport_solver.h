#ifndef REACTIVETRANSPORTGPU_REACTIVE_TRANSPORT_SOLVER_H
#define REACTIVETRANSPORTGPU_REACTIVE_TRANSPORT_SOLVER_H

#include <thread>

#include "equilibrium_solver/minimization_result_info.h"
#include "reactive_transport_solver_abstract.h"

namespace reactive_transport_solver {

using equilibrium_solver::MinimizationResultInfo;

static size_t num_threads_from_env() {
    const char* nthreads_env = std::getenv("NTHREADS");
    if (nthreads_env==NULL)
        throw std::runtime_error("Number of threads unspecified.");
    return std::stoi(nthreads_env);
}

struct ReactiveTransportSolver : ReactiveTransportSolverAbstract {
    ReactiveTransportSolver(ReactiveTransportSolverConf conf_) : ReactiveTransportSolverAbstract(conf_) {}

    std::string identifier() {
        #ifdef USE_MULTIPLE_THREADS
        return "cpu_mt_"+std::to_string(num_threads_from_env());
        #else
        return "cpu";
        #endif
    }

    void equilibration_step() {
        // initialize variables
        size_t ncells = states.size();

        #ifndef USE_MULTIPLE_THREADS
        for (size_t icell=0; icell<ncells; ++icell) {
            component_amounts_t bp = b.row(icell);
            auto result = equilibrate(conf.thermodynamic_properties(), bp, states[icell], conf.minimizer_options, false);
            iterations[icell] = result.it;
        }
        #else
        //size_t nthreads = std::thread::hardware_concurrency();
        static size_t nthreads = num_threads_from_env();

        /*// assign each thread the amount of cells it should process
        std::vector<size_t> work_per_thread(nthreads, ncells/nthreads);
        for (size_t i=0; i<(ncells%nthreads); ++i) {
            work_per_thread[i]+=1;
        }
        // assign each thread a range of cell indices
        std::vector<size_t> thread_cell_indices;
        thread_cell_indices.reserve(nthreads);
        size_t tmp=0;
        thread_cell_indices.push_back(tmp);
        for (size_t i=0; i<nthreads; ++i) {
            tmp += work_per_thread[i];
            thread_cell_indices.push_back(tmp);
        }

        #pragma omp parallel num_threads(nthreads)
        {
            size_t tidx = omp_get_thread_num();
            for (size_t icell=thread_cell_indices[tidx]; icell<thread_cell_indices[tidx+1]; ++icell) {
                component_amounts_t bp = b.row(icell);
                auto result = equilibrate(conf.thermodynamic_properties(), bp, states[icell], conf.minimizer_options, false);
                iterations[icell] = result.it;
            }
        }
        */
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for (size_t icell=0; icell<ncells; ++icell) {
            component_amounts_t bp = b.row(icell);
            auto result = equilibrate(conf.thermodynamic_properties(), bp, states[icell], conf.minimizer_options, false);
            iterations[icell] = result.it;
        }
        #endif
    }
};

}

#endif //REACTIVETRANSPORTGPU_REACTIVE_TRANSPORT_SOLVER_H
