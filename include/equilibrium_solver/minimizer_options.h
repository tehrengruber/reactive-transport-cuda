#ifndef REACTIVETRANSPORTGPU_MINIMIZER_OPTIONS_H
#define REACTIVETRANSPORTGPU_MINIMIZER_OPTIONS_H

namespace equilibrium_solver {

struct MinimizerOptions {
    size_t imax = 100;
    numeric_t mu = 1.0e-14;
    numeric_t tau = 0.99999;
    numeric_t tol = 1.0e-6;
    bool output = false;
    bool output_lse = false;
};

}

#endif //REACTIVETRANSPORTGPU_MINIMIZER_OPTIONS_H
