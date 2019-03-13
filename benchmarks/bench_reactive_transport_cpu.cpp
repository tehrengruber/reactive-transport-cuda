#include <string>
#include "reactive_transport_solver/reactive_transport_solver.h"

using reactive_transport_solver::ReactiveTransportSolverConf;
using reactive_transport_solver::ReactiveTransportSolver;

int main() {
    double mulp_fac = std::pow(10, 1.0/10);
    for (double ncells_=100000; ncells_<=100001; ncells_*=mulp_fac) {
        size_t ncells = ncells_;
        std::cout << "ncells: " << ncells << std::endl;

        ReactiveTransportSolverConf conf;
        conf.ncells = ncells;
        conf.plot = false;
        conf.minimizer_options.tol = 1e15;

        ReactiveTransportSolver solver(conf);

        //solver.eq_solver.measure_launch_overhead(conf.thermodynamic_properties());

        for (size_t i=0; i<10; ++i) {
            solver.step();
        }
        solver.save_statistics("cpu_" + std::to_string(ncells));
    }
}