#ifndef REACTIVETRANSPORTGPU_REACTIVE_TRANSPORT_SOLVER_CUDA_H
#define REACTIVETRANSPORTGPU_REACTIVE_TRANSPORT_SOLVER_CUDA_H

#ifndef NVCC
#error "USE NVCC"
#endif

#include "reactive_transport_solver_abstract.h"
#ifdef USE_CUDA_IMPL_1
#include "equilibrium_solver/equilibrium_solver_cuda_v1.h"
#else
#include "equilibrium_solver/equilibrium_solver_cuda_v2.h"
#endif

namespace reactive_transport_solver {

struct ReactiveTransportSolverCuda : ReactiveTransportSolverAbstract {
    using bs_t = ReactiveTransportSolverAbstract::bs_t;
    #ifdef USE_CUDA_IMPL_1
    using eq_solver_t = equilibrium_solver::EquilibriumSolverCudaV1;
    #else
    using eq_solver_t = equilibrium_solver::EquilibriumSolverCudaV2;
    #endif

    eq_solver_t eq_solver;

    ReactiveTransportSolverCuda() :
            ReactiveTransportSolverCuda(ReactiveTransportSolverConf()) {}

    ReactiveTransportSolverCuda(ReactiveTransportSolverConf conf_) :
            ReactiveTransportSolverAbstract(conf_),
            eq_solver(conf.ncells, conf.minimizer_options) {}

    std::string identifier() {
        std::string id = "gpu";

        #ifdef USE_CUDA_IMPL_1
        id = id + "_v1";
        #else
        id = id + "_v2";
        #endif

        #ifdef USE_SHARED_MEM
        id = id + "_sm";
        #endif

        #ifdef USE_MAGMA
        id = id + "_magma";
        #else
        id = id + "_cublas";
        #endif
        return id;
    }

    virtual void equilibration_step() {
        eq_solver.equilibrate(conf.thermodynamic_properties(), b, states);

        #ifdef USE_CUDA_IMPL_1
        for (size_t i=0; i<conf.ncells; ++i) {
            iterations[i] = 0;
        }
        iterations[0] = eq_solver.iterations;
        #else
        for (size_t i=0; i<conf.ncells; ++i) {
            iterations[i] = eq_solver.info().iterations[i];
        }
        #endif
    }
};

}

#endif //REACTIVETRANSPORTGPU_REACTIVE_TRANSPORT_SOLVER_CUDA_H
