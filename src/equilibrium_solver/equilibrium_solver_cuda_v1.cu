#ifdef USE_CUDA_IMPL_1
#include "equilibrium_solver/equilibrium_solver.h"
#include "equilibrium_solver/equilibrium_solver_cuda_v1.h"
#include "cuda/simple_cuda_profiler.h"

//#include "kernel/minimize_cuda_kernel_inplace.cu"
#include "kernel/minimize_cuda_kernel.cu"

namespace equilibrium_solver {

void EquilibriumSolverCudaV1::equilibrate(ThermodynamicProperties thermo_props, bs_t& bs, states_t& states) {
    SimpleCudaProfiler profiler;
    profiler.add("transfer component amounts H2D");
    profiler.add("transfer states H2D");
    profiler.add("transfer states D2H");
    profiler.add("minimization");
    profiler.initialize();

    prepare_states(thermo_props, states);

    // set number of iterations to zero
    int zero = 0;
    gpuErrchk(cudaMemcpyToSymbol(common_device::minimization_kernel_num_iterations, &zero, sizeof(int)));

    CudaExecutionConfiguration kl_conf("minimize_kernel", (void*) minimize_kernel, ncells);

    // transfer data to device
    profiler.start("transfer states H2D");
    transfer_states_to_device(states);
    profiler.stop("transfer states H2D");
    profiler.start("transfer component amounts H2D");
    transfer_component_amounts_to_device(bs);
    profiler.stop("transfer component amounts H2D");

    // call actual kernel
    profiler.start("minimization");
    //minimize_kernel<<<(ncells+5)/6, 6>>>(device_thermo_props_ptr,
    //        ncells, device_bs_raw_ptr, device_states_x, device_states_y, device_states_z, device_min_opt_ptr);

    minimize_kernel<<<kl_conf.grid_size, kl_conf.block_size>>>(thermo_props,
            ncells, device_bs_raw_ptr, device_states_x, device_states_y, device_states_z, options);
    profiler.stop("minimization");

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // transfer data to host
    profiler.start("transfer states D2H");
    transfer_states_to_host(states);
    profiler.stop("transfer states D2H");

    // output throughput in processed cells/second
    profiler.update();
    //profiler.print();

    finalize_states(thermo_props, states);

    gpuErrchk(cudaMemcpyFromSymbol(&iterations, common_device::minimization_kernel_num_iterations, sizeof(int)));
}

}
#endif