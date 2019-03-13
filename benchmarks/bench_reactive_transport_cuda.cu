#include <string>
#include "reactive_transport_solver/reactive_transport_solver_cuda.h"

using reactive_transport_solver::ReactiveTransportSolverConf;
using reactive_transport_solver::ReactiveTransportSolverCuda;

int main() {
    double mulp_fac = std::pow(2, 1.0/10);
    for (double ncells_=10; ncells_<=200000; ncells_*=mulp_fac) {
        size_t ncells = ncells_;
        std::cout << "ncells: " << ncells << std::endl;

        ReactiveTransportSolverConf conf;
        conf.ncells = ncells;
        conf.plot = false;
        conf.minimizer_options.tol = 0;

        ReactiveTransportSolverCuda solver(conf);

        //solver.eq_solver.measure_launch_overhead(conf.thermodynamic_properties());

        for (size_t i=0; i<10; ++i) {
            solver.step();
        }
        solver.save_statistics(std::to_string(ncells));
    }
}