/*// Import the Python package Numpy so we can perform linear algebra calculations
from numpy import *

// Import copy package to make deep copies of EquilibriumState objects
        import copy

// Import matplotlib for the plots
        import matplotlib.pyplot as plt

// Configure the plot styles
        plt.style.use('ggplot')
plt.rc('font', sizef=16)
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rc('lines', linewidth=4)
plt.rc('figure', autolayout=True)*/

#include <Eigen/Dense>
#include <map>
#include <cassert>
#include <iostream>
#include <vector>
#include <fstream>

#include <matplotlibcpp.h>

#include "profiler/profiler.hpp"
#include "profiler/profiler_io.cpp"

#include "common.h"
#include "equilibrium_solver/equilibrium_solver.h"
#include "equilibrium_solver/cuda_equilibrium_solver.h"
#include "newton_1d.h"
#include "chemistry.h"
#include "simple_timer.h"
#include "string_format.h"
#include "transport.h"

namespace plt = matplotlibcpp;

using namespace common;

using equilibrium_solver::MinimizerOptions;
using equilibrium_solver::EquilibriumState;
using equilibrium_solver::EquilibriumStateSOA;
using equilibrium_solver::equilibrate;
using equilibrium_solver::equilibrate_batch;
using chemistry::ThermodynamicProperties;
using chemistry::elementAmounts;
using chemistry::elementAmountsAqueous;
using chemistry::component_amounts;

#ifdef USE_CUDA
using equilibrium_solver::equilibrate_batch;
#endif

// TODO: check that things are not copied without reason

int main(int argc, char** argv) {
    /*
     * Configuration
     */
    const size_t ncells = argc > 1 ? std::stoi(argv[1]) : 500000;
    numeric_t D  = 0;      // in units of m2/s
    // D  = 1.0e-9      // in units of m2/s
    // v  = 1.0/day     // in units of m/s
    numeric_t v  = 1.0/day;     // in units of m/s

    numeric_t dx = 1.0/ncells;     // in units of m
    // dt = 1*hour   // in units of s
    numeric_t dt = 60*minute;   // in units of s
    // dt = dx**2/D * 0.1  // in units of s

    numeric_t T = 60.0 + 273.15;  // 60 degC in K
    numeric_t P = 100 * 1e5;      // 100 bar in Pa

    // components amounts in the boundary cell and initial amounts in all cells
    //  kgH2O, molCO2, molNaCl, molCaCO3
    auto b_bc = component_amounts(1.0, 0.05, 0.0, 0.0);
    auto b_ic = component_amounts(1.0, 0.0, 0.7, 0.1);

    // initialize variables
    ThermodynamicProperties thermo_props(T, P);
    MinimizerOptions options;

    // initialize output
    if (options.output) {
        std::remove("minimize-output.txt");
    }
    if (options.output_lse) {
        std::remove("lse-output.bin");
        std::remove("rhs-output.bin");
    }

    /*
     * Initialize cells
     */
    EquilibriumState state_bc;
    EquilibriumState state_ic;
    equilibrate(thermo_props, b_bc, state_bc, options, true);
    auto min_res_init = equilibrate(thermo_props, b_ic, state_ic, options, true);

    Eigen::Matrix<numeric_t, Eigen::Dynamic, num_components, Eigen::RowMajor> b0(ncells, num_components);
    b0.setZero();
    Eigen::Matrix<numeric_t, Eigen::Dynamic, num_components, Eigen::RowMajor> b(ncells, num_components);
    b.setZero();
    Eigen::Matrix<numeric_t, Eigen::Dynamic, num_components, Eigen::RowMajor> bfluid(ncells, num_components);
    bfluid.setZero();
    Eigen::Matrix<numeric_t, Eigen::Dynamic, num_components, Eigen::RowMajor> bsolid(ncells, num_components);
    bsolid.setZero();
    std::vector<size_t> iterations(ncells, min_res_init.it);

    for (size_t i=0; i<ncells; ++i) {
        b.row(i) = elementAmounts(state_ic.n);
    }

    // the number of states
    EquilibriumStateSOA states(ncells, state_ic);

    // ylim = None
    numeric_t ylim[2] = {0.0, 1.1};

    Vector<numeric_t, Eigen::Dynamic> x(ncells);
    x.setLinSpaced(ncells, 0., 1.);

    size_t nsteps = 100;
    numeric_t t = 0.0;
    numeric_t equilibration_runtime = 0;
    numeric_t total_iterations = 0;

    //while (true) {
    for (size_t step=0; step<nsteps; ++step) {
        // pH = [-log10(state.a[iH]) for state in states]
        std::vector<numeric_t> nCa(ncells);
        std::vector<numeric_t> nCO2(ncells);
        std::vector<numeric_t> nCaCO3(ncells);
        for (size_t i=0; i<ncells; ++i) {
            nCO2[i] = states[i].n[iCO2];
            nCa[i] = states[i].n[iCa];
            nCaCO3[i] = states[i].n[iCaCO3];
        }
        plt::title(string_format("Time = %.2f day", t/day));
        //if (ylim is not None) {
        //    plt.ylim(ylim[0], ylim[1]);
        //}
        //plt::plot(x, pH);
        plt::subplot(2, 1, 1);
        plt::plot(stdvec_from_eigvec(x), nCO2), make_map({"label", "CO2(aq)"});
        plt::plot(stdvec_from_eigvec(x), nCa), make_map({"label", "Ca++"});
        plt::plot(stdvec_from_eigvec(x), nCaCO3), make_map({"label", "Calcite"});
        plt::ylim(0.0, 0.11);
        plt::subplot(2, 1, 2);
        plt::plot(stdvec_from_eigvec(x), iterations);
        plt::pause(0.5);
        //plt::show(true);
        //plt::legend(make_map("loc", "center right"));
        //if (ylim is None: ylim = plt.ylim())
        //    plt::pause(0.5);
        //}

        // savetxt(output, u, delimiter=',', newline=' ')
        // output.write('\n')

        for (size_t icell=0; icell<ncells; ++icell) {
            bfluid.row(icell) = elementAmountsAqueous(states[icell].n);
            // bsolid[icell] = elementAmountsMineral(states[icell].n)
            bsolid.row(icell) = formula_matrix.col(num_species-1) * states[icell].n[num_species-1];
        }

        PROFILER_START("transport");
        for (size_t j=0; j<num_components; ++j) {
            // todo: check memory layout
            Vector<numeric_t, Eigen::Dynamic> tmp = bfluid.col(j);
            transport(tmp, dt, dx, v, D, b_bc[j]);
            bfluid.col(j) = tmp;
        }
        PROFILER_STOP("transport");

        b = bsolid + bfluid;

        SimpleTimer timer;
        timer.tic();
        #ifdef USE_CUDA
        auto equilibration_result = equilibrate_batch_cuda(thermo_props, b, states, options);
        #else
        auto equilibration_result = equilibrate_batch(thermo_props, b, states, options);
        #endif
        timer.toc();

        // collect some data
        size_t total_iterations_step=0;
        for (size_t i=0; i<ncells; ++i) {
            iterations[i] = equilibration_result[i].it;
            total_iterations_step += equilibration_result[i].it;
            if (!equilibration_result[i].converged) {
                std::cout << i << " did not converge " << std::endl;
            }
        }

        // output
        std::cout << "iterations/s: " << total_iterations_step << " " << total_iterations_step/timer.duration() << std::endl;
        std::cout << "runtime: " << timer.duration() << std::endl;
        std::cout << "Progress:" << (b - b0).norm() << std::endl;

        equilibration_runtime += timer.duration();
        total_iterations += total_iterations_step;
        std::cout << "ti " << total_iterations << std::endl;

        if ((b-b0).norm() < 1e-2)
            break;

        // if t >= dt*nsteps:
        // if linalg.norm(b - b0) < 1e-2:
        //     break

        b0 = b;
        t += dt;

        plt::clf();
    }

    std::ofstream iteration_throughput_f;
    iteration_throughput_f.open("iteration_throughput.txt", std::ios::out | std::ios::app);
    iteration_throughput_f << ncells << " " << total_iterations/equilibration_runtime << std::endl;
    iteration_throughput_f.close();

    std::cout << "Finished!" << std::endl;

    Profiler::IO::print_timings();
}