#include "simple_timer.h"
#include "reactive_transport_solver.h"
#include "reactive_transport_solver_cuda.h"

int main() {
    SimpleTimer t;
    size_t n_repititions = 10;

    // reactive transport solver configuration
    ReactiveTransportSolverConf conf;
    conf.ncells = 1000;
    conf.minimizer_options.tol = 0;

    // setup solver
    ReactiveTransportSolver solver_cpu;
    ReactiveTransportSolverCuda solver_cuda;

    for (size_t i=0; i<n_repititions; ++i) {
        solver_cpu.step();
    }
    std::cout << "cpu throughput" << solver_cpu.average_throughput() << std::endl;

    for (size_t i=0; i<n_repititions; ++i) {
        solver_gpu.step();
    }
    std::cout << "gpu throughput" << solver_gpu.average_throughput() << std::endl;
}