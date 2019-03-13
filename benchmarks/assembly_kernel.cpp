#include "simple_timer.h"
#include "reactive_transport_solver_cuda.h"

int main() {
    SimpleTimer t;
    size_t n_repititions = 10;

    // reactive transport solver configuration
    ReactiveTransportSolverConf conf;
    conf.ncells = 1000000;

    // setup solver
    ReactiveTransportSolverCuda solver_cuda;

    solver_cuda.batch_eq_solver
}