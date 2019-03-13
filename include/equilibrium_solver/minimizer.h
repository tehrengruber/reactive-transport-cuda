#ifndef REACTIVETRANSPORTGPU_MINIMIZER_H
#define REACTIVETRANSPORTGPU_MINIMIZER_H

#include <fstream>

#include "common.h"
#include "common_cuda.h"
#include "profiler/profiler.hpp"

#include "equilibrium_solver/equilibrium_state.h"
#include "equilibrium_solver/optimum_problem.h"
#include "equilibrium_solver/minimizer_options.h"
#include "equilibrium_solver/minimization_result_info.h"

#include "gauss_partial_pivoting.h"

namespace equilibrium_solver {

// make variables on the device visible
#ifdef __CUDA_ARCH__
    using namespace common_device;
#else
    using namespace common;
#endif

// Minimize a function f(x) subject to constraints Ax = b and x >= 0.
template <typename EQUI_STATE_T, typename T>
DEVICE_DECL_SPEC HOST_DECL_SPEC
MinimizationResultInfo minimize(EQUI_STATE_T&& state, const OptimumProblem<T>& problem,
        const MinimizerOptions& opt, bool init=false) {
    const auto imax = opt.imax;
    const auto mu = opt.mu;
    const auto tau = opt.tau;
    const auto tol = opt.tol;

    #if !defined(NDEBUG) && !defined(__CUDA_ARCH__)
    std::fstream output;
    if (opt.output) {
        output = std::fstream("minimize-output.txt", std::ios::app);
    }

    std::ofstream output_lhs;
    std::ofstream output_rhs;

    if (opt.output_lse) {
        output_lhs = std::ofstream("lse-output.bin", std::ios::out | std::ios::binary | std::ios::app);
        output_rhs = std::ofstream("rhs-output.bin", std::ios::out | std::ios::binary | std::ios::app);
    }
    #endif

    const auto& A = problem.A;
    const auto& b = problem.b;

    constexpr size_t m = formula_matrix_t::RowsAtCompileTime;
    constexpr size_t n = formula_matrix_t::ColsAtCompileTime;
    constexpr size_t p = m + n;
    constexpr size_t t = m + 2 * n;

    /*
     * see EquilibriumState constr
     */
    if (init) {
        state.x.setConstant(mu);
        state.y.setConstant(0);
        state.z.setConstant(1);
    }

    auto& x = state.x;
    auto& y = state.y;
    auto& z = state.z;

    #if !defined(NDEBUG) && !defined(__CUDA_ARCH__)
    if (opt.output) {
        std::stringstream header;
        for (size_t i=0; i<header.tellp(); ++i) { // bar
            header << "=";
        }
        header << "\n"
               << std::setw(20) << "Iteration"
               << std::setw(20) << "Error";
        for (size_t i=0; i<n; ++i) {
            header << std::setw(20) << "x[" << species[i] << "]";
        }
        for (size_t i=0; i<m; ++i) {
            header << std::setw(20) << "y[" << components[i] << "]";
        }
        for (size_t i=0; i<n; ++i) {
            header << std::setw(20) << "z[" << species[i] << "]";
        }
        header << "\n";
        size_t header_len = header.tellp();
        for (size_t i=0; i<header_len; ++i) { // bar
            header << "=";
        }
        header << "\n";
        output << header.str();
    }
    #endif

    size_t it;
    numeric_t error;
    for (it=0; it<imax; ++it) {
        Vector<numeric_t, t> F;
        F.setConstant(0);
        Matrix<numeric_t, t, t> J;
        J.setConstant(0);

        // f - scalar
        // g - vector
        // H - matrix
        ST_VER_ONLY(HOST_VER_ONLY(PROFILER_START("equilibrate.obj")));
        chemistry::ObjectiveResult obj_res = problem.objective(x);
        ST_VER_ONLY(HOST_VER_ONLY(PROFILER_STOP("equilibrate.obj")));
        auto& f = obj_res.f;
        auto& g = obj_res.g;
        auto& H = obj_res.H;

        /*HOST_VER_ONLY(PROFILER_START("cache_flush"));
        clear_cpu_cache();
        HOST_VER_ONLY(PROFILER_STOP("cache_flush"));*/

        // Assemble the negative of the residual vector -F
        //     [g(x) - tr(A)*y - z]
        // F = [      A*x - b     ]
        //     [    X*Z*e - mu    ]
        // 2*num_species subs, num_species*(num_components-1 adds and num_components muls)
        ST_VER_ONLY(HOST_VER_ONLY(PROFILER_START("equilibrate.vec_asmbl")));
        F.head(n) = g - A.transpose()*y - z;
        F.segment(n, m) = A*x - b;
        F.tail(n) = (x.array() * z.array()).matrix() - mu*Vector<numeric_t, common::num_species>::Ones();
        ST_VER_ONLY(HOST_VER_ONLY(PROFILER_STOP("equilibrate.vec_asmbl")));

        // Calculate the optimality, feasibility, complementarity errors
        numeric_t error_opt = F.head(n).template lpNorm<Infinity>();
        numeric_t error_fea = F.segment(n, p-n).template lpNorm<Infinity>();
        numeric_t error_com = F.tail(n).template lpNorm<Infinity>();

        // Calculate the current total error
        error = std::max({error_opt, error_fea, error_com});

        //if (!init)
        //    std::cout << "error: " << error << std::endl;

        #if !defined(NDEBUG) && !defined(__CUDA_ARCH__)
        if (opt.output) {
            std::stringstream ss;

            ss << std::setw(20) << it
               << std::setw(20) << error;

            for (size_t i=0; i<n; ++i) {
                ss << std::setw(20) << x[i];
            }
            for (size_t i=0; i<m; ++i) {
                ss << std::setw(20) << y[i];
            }
            for (size_t i=0; i<n; ++i) {
                ss << std::setw(20) << z[i];
            }
            ss << "\n";

            output << ss.str();
        }
        #endif

        // Check if the calculation has converged
        if (error < tol)
            break;

        // Assemble the Jacoabian matrix J
        //     [H -tr(A) -I]
        // J = [A    0    0]
        //     [Z    0    X]
        ST_VER_ONLY(HOST_VER_ONLY(PROFILER_START("equilibrate.mat_asmbl")));
        J.setConstant(0);
        J.block(0,   0,   n,   n) = H;
        J.block(0,   n,   n,   p-n) = -A.transpose();
        J.block(0,   t-n, n,   n).diagonal().setConstant(-1);
        J.block(n,   0,   p-n, n) = A;
        J.block(t-n, 0,   n,   n).diagonal() = z;
        J.block(t-n, t-n, n,   n).diagonal() = x;
        ST_VER_ONLY(HOST_VER_ONLY(PROFILER_STOP("equilibrate.mat_asmbl")));

        Vector<numeric_t, t> delta;

        #if !defined(NDEBUG) && !defined(__CUDA_ARCH__)
        if (opt.output_lse) {
            output_lhs.write((char*) J.data(), J.size() * sizeof(numeric_t));
            output_rhs.write((char*) F.data(), F.size() * sizeof(numeric_t));
        }
        #endif

        ST_VER_ONLY(HOST_VER_ONLY(PROFILER_START("cache_flush")));
        //clear_cpu_cache();
        ST_VER_ONLY(HOST_VER_ONLY(PROFILER_STOP("cache_flush")));

        #if !defined(__CUDA_ARCH__)
        ST_VER_ONLY(HOST_VER_ONLY(PROFILER_START("equilibrate.lu")));
        delta = J.partialPivLu().solve(-F);
        ST_VER_ONLY(HOST_VER_ONLY(PROFILER_STOP("equilibrate.lu")));
        #else
        F = -F;
        gauss<numeric_t, t>(&J(0, 0), &F[0], &delta[0]);
        #endif

        // todo: omit copy
        Eigen::Ref<Vector<numeric_t, n>> dx = delta.head(n);
        Eigen::Ref<Vector<numeric_t, p-n>> dy = delta.segment(n, p-n);
        Eigen::Ref<Vector<numeric_t, n>> dz = delta.tail(n);

        // Calculate the new values for x and z
        ST_VER_ONLY(HOST_VER_ONLY(PROFILER_START("equilibrate.state_var_update")));
        for (size_t i=0; i<n; ++i) {
            x[i] += (x[i] + dx[i] > 0.0) ? dx[i] : (-tau * x[i]);
            z[i] += (z[i] + dz[i] > 0.0) ? dz[i] : (-tau * z[i]);
        }

        // Calculate the new values for y
        y += dy;
        ST_VER_ONLY(HOST_VER_ONLY(PROFILER_STOP("equilibrate.state_var_update")));

        #if !defined(NDEBUG) && !defined(__CUDA_ARCH__)
        if (opt.output) {
            output.flush();
        }
        #endif
    }

    MinimizationResultInfo ret(it, it < imax, error);
    return ret;
}

}

#endif //REACTIVETRANSPORTGPU_MINIMIZER_H
