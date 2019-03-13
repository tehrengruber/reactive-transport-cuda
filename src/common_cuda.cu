#include <cuda_runtime.h>
#include "common.h"
#include "common_cuda.h"

namespace common_device {

// The universal gas constant in units of J/(mol*K)
__constant__ numeric_t R;

// The reference pressure P(ref) (in units of Pa)
__constant__ numeric_t Pref;

__constant__ int slice_aqueous[2];
__constant__ int slice_gaseous[2];
__constant__ int slice_mineral[2];

// Create a list with the charges of the aqueous species
__constant__ int charges[common::num_species_aqueous];

// Create a list with the indices of the charged species
__constant__ int icharged[common::num_charged_species()];

__device__ int minimization_kernel_num_iterations;

// Construct the formula matrix of the chemical system
VAR_MANUELL_INITILIZATION(common::formula_matrix_t, formula_matrix);

VAR_MANUELL_INITILIZATION(common::formula_matrix_trans_t, formula_matrix_trans);

VAR_MANUELL_INITILIZATION(ChemPotTable, u0_table);

// Create an array with the molar masses of the components
VAR_MANUELL_INITILIZATION((Vector<numeric_t, common::num_components>), components_molar_masses);

// Create an array with the molar masses of the species
VAR_MANUELL_INITILIZATION((Vector<numeric_t, common::num_species>), species_molar_masses);

void* initialize() {
    cudaDeviceSynchronize();
    gpuErrchk(cudaMemcpyToSymbol(R, &common::R, sizeof(decltype(R))));
    gpuErrchk(cudaMemcpyToSymbol(Pref, &common::Pref, sizeof(decltype(Pref))));
    gpuErrchk(cudaMemcpyToSymbol(slice_aqueous, &common::slice_aqueous, sizeof(decltype(slice_aqueous))));
    gpuErrchk(cudaMemcpyToSymbol(slice_gaseous, &common::slice_gaseous, sizeof(decltype(slice_gaseous))));
    gpuErrchk(cudaMemcpyToSymbol(slice_mineral, &common::slice_mineral, sizeof(decltype(slice_mineral))));
    gpuErrchk(cudaMemcpyToSymbol(charges, &common::charges, sizeof(decltype(charges))));
    gpuErrchk(cudaMemcpyToSymbol(charges, &common::charges, sizeof(decltype(charges))));
    gpuErrchk(cudaMemcpyToSymbol(icharged, &common::icharged, sizeof(decltype(icharged))));
    gpuErrchk(cudaMemcpyToSymbol(formula_matrix_, &common::formula_matrix, sizeof(decltype(formula_matrix))));
    gpuErrchk(cudaMemcpyToSymbol(u0_table_, &common::u0_table, sizeof(decltype(u0_table))));
    gpuErrchk(cudaMemcpyToSymbol(components_molar_masses_, &common::components_molar_masses, sizeof(decltype(common::components_molar_masses))));
    gpuErrchk(cudaMemcpyToSymbol(species_molar_masses, &common::species_molar_masses, sizeof(decltype(common::species_molar_masses))));

    auto formula_matrix_transposed = common::formula_matrix.transpose();
    gpuErrchk(cudaMemcpyToSymbol(formula_matrix_trans_, &formula_matrix_transposed, sizeof(decltype(formula_matrix_transposed))));

    return nullptr;
}

//void* _dummy = initialize();

}
