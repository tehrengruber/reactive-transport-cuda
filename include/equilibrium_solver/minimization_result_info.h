#ifndef REACTIVETRANSPORTGPU_MINIMIZATION_RESULT_INFO_H
#define REACTIVETRANSPORTGPU_MINIMIZATION_RESULT_INFO_H

#include "common.h"

namespace equilibrium_solver {

struct MinimizationResultInfo {
    size_t it;
    bool converged;
    numeric_t error;

    DEVICE_DECL_SPEC HOST_DECL_SPEC
    MinimizationResultInfo() : it(0), converged(false), error(-1) {}

    DEVICE_DECL_SPEC HOST_DECL_SPEC
    MinimizationResultInfo(size_t it_, bool converged_, numeric_t error_) : it(it_), converged(converged_),
                                                                            error(error_) {}
};

}

#endif //REACTIVETRANSPORTGPU_MINIMIZATION_RESULT_INFO_H
