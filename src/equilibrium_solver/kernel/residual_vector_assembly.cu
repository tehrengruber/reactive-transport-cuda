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
void residual_assembly_kernel(ThermodynamicProperties thermo_props,
                                  MinimizerOptions options,
                                  MinimizationResultInfoCuda& info,
                                  Vector<numeric_t, common::num_species>* xs,
                                  Vector<numeric_t, common::num_components>* ys,
                                  Vector<numeric_t, common::num_species>* zs,
                                  numeric_t* bs_ptr,
                                  numeric_t* Fs) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    constexpr size_t m = formula_matrix_t::RowsAtCompileTime;
    constexpr size_t n = formula_matrix_t::ColsAtCompileTime;
    constexpr size_t p = m + n;
    constexpr size_t t = m + 2 * n;
    const numeric_t mu = options.mu;

    __shared__ numeric_t x_[16*common::num_species];
    __shared__ numeric_t y_[16*common::num_components];
    __shared__ numeric_t z_[16*common::num_species];
    __shared__ numeric_t b_[16*common::num_components];
    __shared__ numeric_t F_[16*t];

    if (idx < info.n_active) {
        int cidx = info.active_indices[idx];

        // initialize variables
        Map<Vector<numeric_t, common::num_species>> x(x_+common::num_species*threadIdx.x);
        Map<Vector<numeric_t, common::num_components>> y(y_+common::num_components*threadIdx.x);
        Map<Vector<numeric_t, common::num_species>> z(z_+common::num_species*threadIdx.x);
        Map<component_amounts_t> b(b_+common::num_components*threadIdx.x);
        Map<Vector<numeric_t, t>> F(F_+t*threadIdx.x);

        // load data
        x = xs[cidx];
        //print("%f %f", x.norm(), xs[cidx].norm());
        y = ys[cidx];
        z = zs[cidx];
        b = Map<component_amounts_t>(bs_ptr+cidx*num_components);

        auto& A = formula_matrix;

        // compute gibbs energy
        //ObjectiveResult obj_res = gibbs_energy(thermo_props, x);
        //auto& g = obj_res.g;
        //auto& H = obj_res.H;

        // assemble vector
        gibbs_energy_opt_inplace_gradient(thermo_props, x, F.head(n));
        F.head(n) -=  A.transpose()*y + z;
        F.segment(n, m) = A*x - b;
        F.tail(n) = (x.array() * z.array()).matrix() - mu*Vector<numeric_t, common::num_species>::Ones();
        F = -F;

        // write back
        Map<Vector<numeric_t, t>>(Fs+cidx*t) = F;

        // calculate error
        numeric_t error = F.template lpNorm<Infinity>();
        info.error[cidx] = error;
    }
}

}