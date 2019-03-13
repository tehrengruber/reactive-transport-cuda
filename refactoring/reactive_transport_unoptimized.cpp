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

namespace plt = matplotlibcpp;

using namespace common;

using equilibrium_solver::MinimizerOptions;
using equilibrium_solver::EquilibriumState;
using equilibrium_solver::equilibrate;
using equilibrium_solver::equilibrate_batch;
using chemistry::ThermodynamicProperties;
using chemistry::elementAmounts;
using chemistry::elementAmountsAqueous;

#ifdef USE_CUDA
using equilibrium_solver::equilibrate_batch;
#endif

// TODO: check that things are not copied without reason

std::string string_format(const std::string fmt, ...) {
    int n, size=100;
    std::string str;
    va_list ap;

    while (1) {
        str.resize(size);
        va_start(ap, fmt);
        int n = vsnprintf(&str[0], size, fmt.c_str(), ap);
        va_end(ap);

        if (n > -1 && n < size)
            return str;
        if (n > -1)
            size = n + 1;
        else
            size *= 2;
    }
}

using keyword_map_t = std::map<std::string, std::string>;

keyword_map_t make_map(std::initializer_list<std::string> list) {
    using it_t = std::initializer_list<std::string>::iterator;
    assert(list.size()%2==0);

    keyword_map_t map;
    for (it_t it=list.begin(); it < list.end(); it+=2) {
        map[*it] = *(it+1);
    }

    return map;
};

#include "gauss_partial_pivoting.h"

// Auxiliary funtion that calculates the vector b of component amounts for
// a given recipe involving H2O, CO2, NaCl, and CaCO3.
component_amounts_t component_amounts(numeric_t kgH2O, numeric_t molCO2, numeric_t molNaCl, numeric_t molCaCO3) {
    numeric_t molH2O = 55.508 * kgH2O;
    component_amounts_t b;
    b << 2*molH2O,                        // H
            molH2O + 2*molCO2 + 3*molCaCO3,  // O
            molCO2 + molCaCO3,               // C
            molNaCl,                         // Na
            molNaCl,                         // Cl
            molCaCO3;                        // Ca
    // Ensure the amounts of each component is greater than zero
    for (size_t i=0; i<b.size(); ++i) {
        if (b[i] <= 0.0)
            b[i] = 1e-10;  // replace zero or negative by a small positive number
    }
    return b;
}

// Auxiliary time related constants
const numeric_t second = 1.0;
const numeric_t minute = 60.0;
const numeric_t hour = 60 * minute;
const numeric_t day = 24 * hour;
const numeric_t year = 365 * day;

// Solve a tridiagonal matrix equation using Thomas algorithm.
// TODO: references?
Vector<numeric_t, Eigen::Dynamic> thomas(Vector<numeric_t, Eigen::Dynamic>& a, Vector<numeric_t, Eigen::Dynamic>& b,
                            Vector<numeric_t, Eigen::Dynamic>& c, Vector<numeric_t, Eigen::Dynamic>& d) {
    size_t n = d.size();
    c[0] /= b[0];
    for (size_t i=1; i < n-1; ++i) {
        c[i] /= b[i] - a[i]*c[i - 1];
    }
    d[0] /= b[0];
    for (size_t i=1; i<n; ++i) {
        d[i] = (d[i] - a[i]*d[i - 1])/(b[i] - a[i]*c[i - 1]);
    }
    auto& x = d;
    for (int i=n-2; i>=0; --i) {
        x[i] -= c[i]*x[i + 1];
    }
    return x;
}

// Perform a transport step
void transport(Vector<numeric_t, Eigen::Dynamic>& u, numeric_t dt, numeric_t dx, numeric_t v, numeric_t D, numeric_t ul) {
    size_t ncells = u.size();

    numeric_t alpha = D*dt/(dx*dx);
    numeric_t beta = v*dt/dx;
    Vector<numeric_t, Eigen::Dynamic> a(ncells);
    a.setConstant(-beta - alpha);
    Vector<numeric_t, Eigen::Dynamic> b(ncells);
    b.setConstant(1 + beta + 2*alpha);
    Vector<numeric_t, Eigen::Dynamic> c(ncells);
    c.setConstant(-alpha);
    // b[0] = 1.0
    // c[0] = 0.0
    // u[0] = ul
    u[0] += beta*ul;
    b[0] = 1 + beta + alpha;
    b[ncells-1] = 1 + beta + alpha;
    thomas(a, b, c, u);
}

template <int N>
std::vector<numeric_t> stdvec_from_eigvec(const Vector<numeric_t, N> v) {
    return std::vector<numeric_t>(&v[0], &v[0]+v.size());
}

int main(int argc, char** argv) {
    /*
     * Configuration
     */
    const size_t ncells = argc > 1 ? std::stoi(argv[1]) : 100000;
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
    std::vector<EquilibriumState> states(ncells, state_ic);

    // ylim = None
    numeric_t ylim[2] = {0.0, 1.1};

    Vector<numeric_t, Eigen::Dynamic> x(ncells);
    x.setLinSpaced(ncells, 0., 1.);

    size_t nsteps = 10;
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
        PROFILER_START("equilibrate");
        #ifdef USE_CUDA
        auto equilibration_result = equilibrate_batch_cuda(thermo_props, b, states, options);
        #else
        auto equilibration_result = equilibrate_batch(thermo_props, b, states, options);
        #endif
        PROFILER_STOP("equilibrate");
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