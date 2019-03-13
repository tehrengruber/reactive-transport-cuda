#ifndef REACTIVETRANSPORTGPU_OPTIMUM_PROBLEM_H
#define REACTIVETRANSPORTGPU_OPTIMUM_PROBLEM_H

#include "common.h"

namespace equilibrium_solver {

using common::formula_matrix_t;
using common::component_amounts_t;

template <typename F>
struct OptimumProblem {
    const F objective;
    const formula_matrix_t& A;
    const component_amounts_t& b;

    HOST_DECL_SPEC DEVICE_DECL_SPEC OptimumProblem(F obj_, const formula_matrix_t& A_, const component_amounts_t& b_) : objective(obj_), A(A_), b(b_) {}
};

}

#endif //REACTIVETRANSPORTGPU_OPTIMUM_PROBLEM_H
