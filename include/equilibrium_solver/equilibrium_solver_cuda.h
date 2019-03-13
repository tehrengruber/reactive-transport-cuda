#ifndef REACTIVETRANSPORTGPU_EQUILIBRIUM_SOLVER_CUDA_H
#define REACTIVETRANSPORTGPU_EQUILIBRIUM_SOLVER_CUDA_H

#include <memory>
#include <cublas_v2.h>

#include "common.h"
#include "common_cuda.h"
#include "cuda/simple_cuda_profiler.h"
#include "equilibrium_solver/minimization_result_info_cuda.h"
#include "equilibrium_solver/equilibrium_state.h"
#include "equilibrium_solver/minimization_result_info.h"
#include "chemistry.h"

namespace equilibrium_solver {

using namespace chemistry;

struct EquilibriumSolverCudaAbstract {
    using bs_t = Eigen::Matrix<numeric_t, Eigen::Dynamic, common::num_components, Eigen::RowMajor>;
    //using states_t = std::vector<EquilibriumState>;
    using states_t = EquilibriumStateSOA;

    size_t ncells; // number of cells

    MinimizerOptions options;

    //
    // device data
    //
    Vector<numeric_t, common::num_species>*    device_states_x; // alias to amounts of species in mol
    Vector<numeric_t, common::num_components>* device_states_y; // Lagrange multipliers y in J/mol
    Vector<numeric_t, common::num_species>*    device_states_z; // Lagrange multipliers z in J/mol

    numeric_t* device_bs_raw_ptr;

    EquilibriumSolverCudaAbstract(size_t ncells_, MinimizerOptions options_) : ncells(ncells_),
            options(options_) {
        common_device::initialize();
        // allocate device memory
        if (cudaMalloc((void**) &device_states_x, ncells*sizeof(Vector<numeric_t, common::num_species>)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_states_y, ncells*sizeof(Vector<numeric_t, common::num_components>)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_states_z, ncells*sizeof(Vector<numeric_t, common::num_species>)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
        if (cudaMalloc((void**) &device_bs_raw_ptr, ncells*num_components*sizeof(numeric_t)) != cudaSuccess)
            throw std::runtime_error("Device memory allocation failed");
    }

    EquilibriumSolverCudaAbstract(const EquilibriumSolverCudaAbstract&) = delete;
    EquilibriumSolverCudaAbstract& operator=(const EquilibriumSolverCudaAbstract&) = delete;

    ~EquilibriumSolverCudaAbstract() {
        //cudaFree(device_states_ptr);
        cudaFree(device_states_x);
        cudaFree(device_states_y);
        cudaFree(device_states_z);
        cudaFree(device_bs_raw_ptr);
    }

    void transfer_states_to_device(states_t& states) {
        if (cudaMemcpy((void*) device_states_x, (void*) states.x.data(), ncells*sizeof(Vector<numeric_t, common::num_species>), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("Device memory transfer");
        }
        if (cudaMemcpy((void*) device_states_y, (void*) states.y.data(), ncells*sizeof(Vector<numeric_t, common::num_components>), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("Device memory transfer");
        }
        if (cudaMemcpy((void*) device_states_z, (void*) states.z.data(), ncells*sizeof(Vector<numeric_t, common::num_species>), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("Device memory transfer");
        }
    }

    void transfer_component_amounts_to_device(bs_t& bs) {
        if (cudaMemcpy((void*) device_bs_raw_ptr, (void*) bs.data(), ncells*num_components*sizeof(numeric_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("Device memory transfer");
        }
    }

    void transfer_states_to_host(states_t& states) {
        if (cudaMemcpy((void*) states.x.data(), (void*) device_states_x, ncells*sizeof(Vector<numeric_t, common::num_species>), cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("Device memory transfer");
        }
        if (cudaMemcpy((void*) states.y.data(), (void*) device_states_y, ncells*sizeof(Vector<numeric_t, common::num_components>), cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("Device memory transfer");
        }
        if (cudaMemcpy((void*) states.z.data(), (void*) device_states_z, ncells*sizeof(Vector<numeric_t, common::num_species>), cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("Device memory transfer");
        }
    }

    void prepare_states(ThermodynamicProperties thermo_props, states_t& states) {
        /*for (size_t i=0; i<states.size(); ++i) {
            states[i].y /= R*thermo_props.T;
            states[i].z /= R*thermo_props.T;
        }*/
    }

    void finalize_states(ThermodynamicProperties thermo_props, states_t& states) {
        /*for (size_t i=0; i<states.size(); ++i) {
            numeric_t T = thermo_props.T;
            numeric_t P = thermo_props.P;

            states[i].m = chemistry::masses(states[i].x);
            states[i].y = states[i].y * R * T;
            states[i].z = states[i].z * R * T;
            states[i].a = chemistry::ln_activities(T, P, states[i].x).array().exp().matrix();
            states[i].c = chemistry::concentrations(T, P, states[i].x);
            states[i].g = (states[i].a.array()/states[i].c.array()).matrix();
        }*/
    }

    void equilibrate(ThermodynamicProperties thermo_props, bs_t& bs, states_t& states);
};

}

#endif //REACTIVETRANSPORTGPU_CUDABATCHEQUILIBRATIONSOLVER_H
