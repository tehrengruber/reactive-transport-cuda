#ifndef REACTIVETRANSPORTGPU_NEWTON_1D_H
#define REACTIVETRANSPORTGPU_NEWTON_1D_H

#include <iostream>
#include "common.h"

// Define a function that solves the nonlinear equation f(x) = 0.
// Parameters:
// - f is the function
// - fprime is the function first0order derivative
// - x0 is the initial guess
// Return: the value of x such that f(x) = 0
template <typename F, typename D>
DEVICE_DECL_SPEC HOST_DECL_SPEC
numeric_t newton(F f, D fprime, numeric_t x0) {
    size_t maxiters = 100; // maximum number of iterations
    numeric_t tolerance = 1e-4; // the tolerance for the convergence
    numeric_t x = x0; // start with the solution x being the initial guess
    size_t c=0;
    // Perform one or more Newton iterations
    for (; c<maxiters; ++c) {
        x = x - f(x) / fprime(x); // calculate the new approximation for x
        if (std::abs(f(x)) < tolerance) // check for convergence
            return x; // return x if the calculation converged
    }
    // Raise an error if the calculation did not converge.
#if !defined(NODEBUG) && !defined(__CUDA_ARCH__)
    std::cerr << "Could not calculate the solution of the nonlinear equation in "
              << c << " iterations.";
    throw std::runtime_error("some description");
#endif
    return std::numeric_limits<numeric_t>::signaling_NaN();
}

#endif //REACTIVETRANSPORTGPU_NEWTON_1D_H
