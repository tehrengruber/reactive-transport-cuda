#ifndef REACTIVETRANSPORTGPU_EQUILIBRIUM_SOLVER_CUDA_H
#define REACTIVETRANSPORTGPU_EQUILIBRIUM_SOLVER_CUDA_H

#include "common.h"
#include "common_cuda.h"
#include "chemistry.h"
#include "equilibrium_solver/equilibrium_solver.h"
#include "equilibrium_solver/minimizer.h"

namespace equilibrium_solver {

using namespace chemistry;

std::vector<MinimizationResultInfo> equilibrate_batch_cuda(ThermodynamicProperties& thermo_props,
                                                           Eigen::Matrix<numeric_t, Eigen::Dynamic, common::num_components, Eigen::RowMajor>& bs,
                                                           EquilibriumStateSOA& states,
                                                           MinimizerOptions options, bool init=false);

}
#endif //REACTIVETRANSPORTGPU_EQUILIBRIUM_SOLVER_CUDA_H
