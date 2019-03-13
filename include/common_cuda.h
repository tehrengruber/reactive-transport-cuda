#if !defined(REACTIVETRANSPORTGPU_COMMON_CUDA_H) && defined(__NVCC__)
#define REACTIVETRANSPORTGPU_COMMON_CUDA_H

#include <cuda_runtime.h>
#include "common.h"

template<typename T> struct argument_type;
template<typename T, typename U> struct argument_type<T(U)> { typedef U type; };

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

template <size_t SIZE>
struct Dummy {
    char storage[SIZE];
};

//#define VAR_MANUELL_INITILIZATION(t, var_name) \
//DEVICE_DECL_SPEC __constant__ char __align__(16) var_name##_[sizeof(argument_type<void(t)>::type)]; \
//DEVICE_DECL_SPEC __constant__ argument_type<void(t)>::type& var_name(*reinterpret_cast< argument_type<void(t)>::type*>(var_name##_));

#define VAR_MANUELL_INITILIZATION(t, var_name) \
__constant__ Dummy<sizeof(argument_type<void(t)>::type)> __align__(16) var_name##_; \
__constant__ argument_type<void(t)>::type& var_name(*reinterpret_cast< argument_type<void(t)>::type*>(&var_name##_));

struct CudaExecutionConfiguration {
    int grid_size;
    int block_size;

    CudaExecutionConfiguration(std::string name, void* kernel, int n) {
        int minGridSize;
        cudaOccupancyMaxPotentialBlockSize(
                &minGridSize,
                &block_size,
                (void*)kernel,
                0,
                n);
        grid_size = (n + block_size - 1) / block_size;

        //compute_occupancy(name, kernel);
    }

    CudaExecutionConfiguration(std::string name, void* kernel, int n, int block_size_) {
        block_size = block_size_;
        grid_size = (n+(block_size-1))/block_size;

        //compute_occupancy(name, kernel);
    }

    void compute_occupancy(std::string name, void* kernel) {
        int num_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks,
            (void*) kernel,
            block_size,
            0);

        int device;
        cudaDeviceProp prop;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);

        //std::cout << "Occupancy " << name << " : " << double(num_blocks * block_size)/prop.maxThreadsPerMultiProcessor << std::endl;
    }
};

namespace common_device {

using formula_matrix_t = common::formula_matrix_t;
using gradient_t = common::gradient_t;
using hessian_t = common::hessian_t;
using component_amounts_t = common::component_amounts_t;

// The universal gas constant in units of J/(mol*K)
extern __constant__ numeric_t R;

// The reference pressure P(ref) (in units of Pa)
extern __constant__ numeric_t Pref;

// The number of components and species
using common::num_components;
using common::num_species;

// The number of aqueous, gaseous, and mineral species
using common::num_species_gaseous;
using common::num_species_mineral;
using common::num_species_aqueous;

extern __constant__ int slice_aqueous[2];
extern __constant__ int slice_gaseous[2];
extern __constant__ int slice_mineral[2];

// Create a list with the charges of the aqueous species
extern __constant__ int charges[common::num_species_aqueous];

// Store the index of H2O(l)
using common::iH2O;

// Store the index of H+(aq)
using common::iH;

// Store the index of CO2(aq)
using common::iCO2;

using common::iCa;

using common::iCaCO3;

// Create a list with the indices of the charged species
extern __constant__ int icharged[common::num_charged_species()];

// Construct the formula matrix of the chemical system
extern __constant__ common::formula_matrix_t& formula_matrix;
//extern __constant__ common::formula_matrix_trans_t& formula_matrix_trans;

extern __constant__ ChemPotTable& u0_table;

// Create an array with the molar masses of the components
extern __constant__ Vector<numeric_t, common::num_components>& components_molar_masses;

// Create an array with the molar masses of the species
extern __constant__ Vector<numeric_t, common::num_species>& species_molar_masses;

extern __device__ int minimization_kernel_num_iterations;

// A global variable to allow one to choose if ideal activity models are used
using common::useideal;

// A global variable to allow one to choose only reference state standard
// chemical potentials are used, without interpolation
using common::userefpotentials;

void* initialize();

extern void* _dummy;

}

#endif //REACTIVETRANSPORTGPU_COMMON_CUDA_H
