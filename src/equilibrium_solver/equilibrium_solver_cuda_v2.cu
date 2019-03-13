#include "equilibrium_solver/equilibrium_solver.h"
#include "equilibrium_solver/equilibrium_solver_cuda_v2.h"

#include "kernel/gibbs_energy_kernel.cu"
//#include "kernel/minimization_assembly_kernel.cu"
#include "kernel/minimization_assembly_kernel_fused.cu"
#include "kernel/minimization_assembly_sm_kernel_fused.cu"
#include "kernel/state_update_kernel.cu"

#include "../kernel/gauss_partial_pivoting_kernel.cu"

#ifdef USE_MAGMA
#include "magma_v2.h"
#endif

namespace equilibrium_solver {

void EquilibriumSolverCudaV2::equilibrate(ThermodynamicProperties thermo_props, bs_t& bs, states_t& states) {
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    info().reset();

    constexpr size_t t = 2*num_species+num_components;

    prepare_states(thermo_props, states);

    // setup lu solver
    cublasStatus_t (*cublas_getrf_fptr)(cublasHandle_t,int,numeric_t*[],int,int*,int*,int);
    if (std::is_same<numeric_t, double>::value)
        cublas_getrf_fptr = reinterpret_cast<decltype(cublas_getrf_fptr)>(cublasDgetrfBatched);
    else if (std::is_same<numeric_t, float>::value)
        cublas_getrf_fptr = reinterpret_cast<decltype(cublas_getrf_fptr)>(cublasDgetrfBatched);
    else
        throw std::runtime_error("invalid configuration numeric_t must be one of double, float");

    cublasStatus_t (*cublas_getrs_fptr)(cublasHandle_t,cublasOperation_t,int,int,const numeric_t*[],int lda,const int*,numeric_t *[],int,int*,int);
    if (std::is_same<numeric_t, double>::value)
        cublas_getrs_fptr = reinterpret_cast<decltype(cublas_getrs_fptr)>(cublasDgetrsBatched);
    else if (std::is_same<numeric_t, float>::value)
        cublas_getrs_fptr = reinterpret_cast<decltype(cublas_getrs_fptr)>(cublasSgetrsBatched);
    else
        throw std::runtime_error("invalid configuration numeric_t must be one of double, float");

    // transfer data to the device
    profiler.start("transfer component amounts H2D");
    transfer_component_amounts_to_device(bs);
    profiler.stop("transfer component amounts H2D");

    profiler.start("transfer states H2D");
    transfer_states_to_device(states);
    profiler.stop("transfer states H2D");

    size_t iterations = 0;

    while (!info().finished && iterations < options.imax) {
        // determine launch configurations
        CudaExecutionConfiguration conf_gec("gibbs_energy_kernel", (void*) gibbs_energy_kernel, info().n_active);
        //CudaExecutionConfiguration conf_ass((void*) minimization_assembly_kernel_fused, (info().n_active+3)/4);
        CudaExecutionConfiguration conf_eqsu("state_update_kernel", (void*) state_update_kernel, info().n_active);

        // compute gibbs energy
        /*profiler.start("gibbs energy kernel");
        gibbs_energy_kernel<<<conf_gec.grid_size, conf_gec.block_size>>>(thermo_props, info().n_active, info().active_indices,
                device_states_ptr, device_objective_evals);
        profiler.stop("gibbs energy kernel");*/

        // assemble matrices
        profiler.start("assembly");
        #ifdef USE_SHARED_MEM
        CudaExecutionConfiguration conf_assf("minimization_assembly_sm_kernel_fused", (void*) minimization_assembly_sm_kernel_fused, info().n_active, 16);
        minimization_assembly_sm_kernel_fused<<<conf_assf.grid_size, conf_assf.block_size>>>(thermo_props,
                options, info(), device_states_x, device_states_y, device_states_z, device_bs_raw_ptr, device_Js_ptr, device_Fs_ptr);
        #else
        CudaExecutionConfiguration conf_assf("minimization_assembly_kernel_fused", (void*) minimization_assembly_kernel_fused, info().n_active);
        minimization_assembly_kernel_fused<<<conf_assf.grid_size, conf_assf.block_size>>>(thermo_props, options, info(),
                device_states_x, device_states_y, device_states_z, device_bs_raw_ptr, device_Js_ptr, device_Fs_ptr);
        #endif
        profiler.stop("assembly");

        //std::cout << info().n_active << std::endl;

        /*profiler.start("assembly");
        CudaExecutionConfiguration conf_assf("minimization_assembly_sm_kernel_fused", (void*) minimization_assembly_sm_kernel_fused, info().n_active, 32);
        minimization_assembly_sm_kernel_fused<<<conf_assf.grid_size, conf_assf.block_size>>>(thermo_props,
                options, info(), device_states_x, device_states_y, device_states_z, device_bs_raw_ptr, device_Js_ptr, device_Fs_ptr);

        //minimization_assembly_kernel_fused<<<conf_assf.grid_size, conf_assf.block_size>>>(thermo_props,
        //        options, info(), device_states_x, device_states_y, device_states_z, device_bs_raw_ptr, device_Js_ptr, device_Fs_ptr);
        profiler.stop("assembly");*/

        gpuErrchk( cudaDeviceSynchronize() );

        // update minimization info
        //  here we determine which problems have converged and prepare the input data of the lse solver
        profiler.start("minimization update");
        //info().send_to_host();
        {
            info().update();
            for (size_t i=0; i<info().n_active; ++i) {
                size_t cidx = info().active_indices[i];
                Js_ptrs[i] = device_Js_ptr+cidx*t*t;
                Fs_ptrs[i] = device_Fs_ptr+cidx*t;
            }
            if (cudaMemcpy((void*) device_Js_ptrs, (void*) Js_ptrs, info().n_active*sizeof(numeric_t*), cudaMemcpyHostToDevice) != cudaSuccess) {
                throw std::runtime_error("Device memory transfer");
            }
            if (cudaMemcpy((void*) device_Fs_ptrs, (void*) Fs_ptrs, info().n_active*sizeof(numeric_t*), cudaMemcpyHostToDevice) != cudaSuccess) {
                throw std::runtime_error("Device memory transfer");
            }
            /*CudaExecutionConfiguration conf((void*) minimization_info_update_kernel, ncells/min_info_batch_size);
            minimization_info_update_kernel<<<conf.grid_size, conf.block_size>>>(info(), options,
                    device_Js_ptr, device_Fs_ptr, device_Js_ptrs, device_Fs_ptrs);*/

        }

        if (info().n_active==0)
            break;
        //info().send_to_gpu();
        profiler.stop("minimization update");

        // perform LU factorization
        profiler.start("factorization");
        #ifndef USE_MAGMA
        //CudaExecutionConfiguration conf_fac((void*) gauss_partial_pivoting_kernel, ncells);
        //gauss_partial_pivoting_kernel<<<conf_fac.grid_size, conf_fac.block_size>>>(ncells, device_Js_ptr, device_Fs_ptr, device_Fs_ptr);
        int fac_res = cublas_getrf_fptr(handle, t, device_Js_ptrs, t, device_pivot_arr, device_info_arr, info().n_active);
        if (fac_res != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("LU factorization failed");

        // solve using LU factorization
        int info_res;
        int sol_res = cublas_getrs_fptr(handle, CUBLAS_OP_N, t, 1, const_cast<const numeric_t **>(device_Js_ptrs),
                                        t, device_pivot_arr, device_Fs_ptrs, t, &info_res, info().n_active);
        if (sol_res != CUBLAS_STATUS_SUCCESS) {
            std::cerr << sol_res << std::endl;
            throw std::runtime_error("LU solve failed");
        }
        #else
        for (size_t i=0; i<info().n_active; i+=50000) {
            size_t count = std::min<size_t>(info().n_active-i, 50000);
            magma_dgesv_batched(t, 1, device_Js_ptrs+i, t, magma_pivot_arr_ptrs+i, device_Fs_ptrs+i, t, device_info_arr+i, count, magma_queue);
        }
        #endif
        profiler.stop("factorization");

        // update equilibrium states
        profiler.start("state update");
        state_update_kernel<<<conf_eqsu.grid_size, conf_eqsu.block_size>>>(thermo_props, options, info(),
                device_states_x, device_states_y, device_states_z, device_Fs_ptr);
        profiler.stop("state update");
        // wait until all kernels finished
        gpuErrchk( cudaDeviceSynchronize() );
        profiler.update();
        iterations++;
    }

    //profiler.print();

    // transfer data from device to host
    profiler.start("transfer states D2H");
    transfer_states_to_host(states);
    profiler.stop("transfer states D2H");

    profiler.update();

    // finalize states (i.e. compute activities, concentrations, ...)
    finalize_states(thermo_props, states);
}

}