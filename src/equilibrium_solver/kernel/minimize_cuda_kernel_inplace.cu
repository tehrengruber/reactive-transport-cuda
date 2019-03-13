#include "common.h"
#include "common_cuda.h"
#include "chemistry.h"
#include "equilibrium_solver/minimizer_options.h"
#include "equilibrium_solver/equilibrium_state.h"
#include "equilibrium_solver/equilibrium_solver.h"

namespace equilibrium_solver {

// make variables on the device visible
#ifdef __CUDA_ARCH__
    using namespace common_device;
#else
    using namespace common;
#endif

using chemistry::ThermodynamicProperties;

__global__
void minimize_kernel_inplace(ThermodynamicProperties* thermo_props_ptr,
                     size_t ncells,
                     numeric_t* bs_raw,
                     Vector<numeric_t, common::num_species>* xs,
                     Vector<numeric_t, common::num_components>* ys,
                     Vector<numeric_t, common::num_species>* zs,
                     MinimizerOptions* options_ptr) {
    constexpr size_t m = formula_matrix_t::RowsAtCompileTime;
    constexpr size_t n = formula_matrix_t::ColsAtCompileTime;
    constexpr size_t p = m + n;
    constexpr size_t t = m + 2 * n;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    ThermodynamicProperties& thermo_props(*thermo_props_ptr);
    MinimizerOptions& options(*options_ptr);


    if (idx < ncells) {
        auto& x = xs[idx];
        auto& y = ys[idx];
        auto& z = zs[idx];

        const auto& A = formula_matrix;
        Eigen::Map<component_amounts_t> b(bs_raw+num_components*idx);

        const auto imax = options.imax;
        const auto mu = options.mu;
        const auto tau = options.tau;
        const auto tol = options.tol;

        size_t it;
        numeric_t error;
        for (it=0; it<imax; ++it) {
            Vector<numeric_t, t> F;
            // Assemble the negative of the residual vector -F
            //     [g(x) - tr(A)*y - z]
            // F = [      A*x - b     ]
            //     [    X*Z*e - mu    ]
            // 2*num_species subs, num_species*(num_components-1 adds and num_components muls)
            gibbs_energy_opt_inplace_gradient(thermo_props, x, F.head(n));
            F.head(n) -= A.transpose()*y + z;
            F.segment(n, m) = A*x - b;
            F.tail(n) = (x.array() * z.array()).matrix() - mu*Vector<numeric_t, common::num_species>::Ones();
            F = -F;

            // Calculate the current total error
            numeric_t error = F.template lpNorm<Infinity>();

            // Check if the calculation has converged
            if (error < tol)
                break;

            Matrix<numeric_t, t, t> J;
            J.setConstant(0);

            // Assemble the Jacoabian matrix J
            //     [H -tr(A) -I]
            // J = [A    0    0]
            //     [Z    0    X]
            gibbs_energy_opt_inplace_hessian(thermo_props, x, J.block(0,   0,   n,   n));
            J.block(0,   n,   n,   p-n) = -A.transpose();
            J.block(0,   t-n, n,   n).diagonal().setConstant(-1);
            J.block(n,   0,   p-n, n) = A;
            J.block(t-n, 0,   n,   n).diagonal() = z;
            J.block(t-n, t-n, n,   n).diagonal() = x;

            // solve lse
            Vector<numeric_t, t> delta;
            gauss<numeric_t, t>(&J(0, 0), &F[0], &delta[0]);

            Eigen::Ref<Vector<numeric_t, n>> dx = delta.head(n);
            Eigen::Ref<Vector<numeric_t, p-n>> dy = delta.segment(n, p-n);
            Eigen::Ref<Vector<numeric_t, n>> dz = delta.tail(n);

            // Calculate the new values for x and z
            for (size_t i=0; i<n; ++i) {
                x[i] += (x[i] + dx[i] > 0.0) ? dx[i] : (-tau * x[i]);
                z[i] += (z[i] + dz[i] > 0.0) ? dz[i] : (-tau * z[i]);
            }

            // Calculate the new values for y
            y += dy;
        }
        atomicAdd(&common_device::minimization_kernel_num_iterations, it);
    }
}

}