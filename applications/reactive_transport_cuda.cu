#include <string>
#include <ctime>

#include "reactive_transport_solver/reactive_transport_solver_cuda.h"

using reactive_transport_solver::ReactiveTransportSolverConf;
using reactive_transport_solver::ReactiveTransportSolverCuda;

int main() {
    size_t ncells = 100000;
    std::cout << "ncells: " << ncells << std::endl;

    ReactiveTransportSolverConf conf;
    conf.ncells = ncells;
    conf.plot = false;

    ReactiveTransportSolverCuda solver(conf);

    while (solver.progress > 1e-2) {
        solver.step();
    }

    solver.save_statistics("real_"+std::to_string(time(nullptr)));
}