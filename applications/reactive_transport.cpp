#define SKIP_PROFILING
#include "reactive_transport_solver/reactive_transport_solver.h"

using reactive_transport_solver::ReactiveTransportSolverConf;
using reactive_transport_solver::ReactiveTransportSolver;

int main() {
    ReactiveTransportSolverConf conf;
    conf.ncells = 100000;
    conf.plot = false;

    ReactiveTransportSolver solver(conf);

    while (solver.progress > 1e-2) {
        solver.step();
        solver.save_statistics("real");
    }
}