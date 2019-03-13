#ifndef REACTIVETRANSPORTGPU_EQUILIBRIUM_SOLVER_CUDA_V1_H
#define REACTIVETRANSPORTGPU_EQUILIBRIUM_SOLVER_CUDA_V1_H

#include <memory>
#include <cublas_v2.h>

#include "common.h"
#include "common_cuda.h"
#include "cuda/simple_cuda_profiler.h"
#include "equilibrium_solver/minimization_result_info_cuda.h"
#include "equilibrium_solver/equilibrium_state.h"
#include "equilibrium_solver/minimization_result_info.h"
#include "equilibrium_solver/equilibrium_solver_cuda.h"
#include "chemistry.h"

namespace equilibrium_solver {

using namespace chemistry;

struct EquilibriumSolverCudaV1 : EquilibriumSolverCudaAbstract {
    using parent_t = EquilibriumSolverCudaAbstract;
    using bs_t = parent_t::bs_t;
    using states_t = parent_t::states_t;

    int iterations = 0;

    EquilibriumSolverCudaV1(size_t ncells_, MinimizerOptions options_) : parent_t(ncells_, options_) {}

    virtual void equilibrate(ThermodynamicProperties thermo_props, bs_t& bs, states_t& states);
};

}
#endif //REACTIVETRANSPORTGPU_EQUILIBRIUM_SOLVER_CUDA_V1_H
