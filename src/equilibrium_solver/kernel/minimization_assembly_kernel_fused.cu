#include "common.h"
#include "common_cuda.h"
#include "chemistry.h"
#include "equilibrium_solver/minimizer_options.h"
#include "equilibrium_solver/equilibrium_state.h"
#include "equilibrium_solver/minimization_result_info_cuda.h"

namespace equilibrium_solver {

// make variables on the device visible
#ifdef __CUDA_ARCH__
using namespace common_device;
#else
using namespace common;
#endif

using chemistry::ThermodynamicProperties;

__global__
void minimization_assembly_kernel_fused(ThermodynamicProperties thermo_props,
                                  MinimizerOptions options,
                                  MinimizationResultInfoCuda& info,
                                  Vector<numeric_t, common::num_species>* xs,
                                  Vector<numeric_t, common::num_components>* ys,
                                  Vector<numeric_t, common::num_species>* zs,
                                  numeric_t* bs_ptr,
                                  numeric_t* Js,
                                  numeric_t* Fs) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    constexpr size_t m = formula_matrix_t::RowsAtCompileTime;
    constexpr size_t n = formula_matrix_t::ColsAtCompileTime;
    constexpr size_t p = m + n;
    constexpr size_t t = m + 2 * n;
    const numeric_t mu = options.mu;

    if (idx < info.n_active) {
        int cidx = info.active_indices[idx];

        // initialize variables
        Map<component_amounts_t> b(bs_ptr+cidx*num_components);

        auto& A = formula_matrix;

        auto& x = xs[cidx];
        auto& y = ys[cidx];
        auto& z = zs[cidx];

        // compute gibbs energy
        ObjectiveResult obj_res = gibbs_energy(thermo_props, x);
        auto& g = obj_res.g;
        auto& H = obj_res.H;

        // assemble vector
        Map<Vector<numeric_t, t>> F(Fs+cidx*t);
        F.head(n) = g - A.transpose()*y - z;
        F.segment(n, m) = A*x - b;
        F.tail(n) = (x.array() * z.array()).matrix() - mu*Vector<numeric_t, common::num_species>::Ones();
        F = -F;

        // calculate error
        numeric_t error = F.template lpNorm<Infinity>();
        info.error[cidx] = error;
        info.converged[cidx] = error < options.tol;

        // assemble matrix
        Map<Matrix<numeric_t, t, t>> J(Js+cidx*t*t);
        if (!info.converged[cidx]) {
            J.setConstant(0);
            J.block(0,   0,   n,   n) = H;
            J.block(0,   n,   n,   p-n) = -A.transpose();
            J.block(0,   t-n, n,   n).diagonal().setConstant(-1);
            J.block(n,   0,   p-n, n) = A;
            J.block(t-n, 0,   n,   n).diagonal() = z;
            J.block(t-n, t-n, n,   n).diagonal() = x;
            ++info.iterations[cidx];
        }
    }
}

}