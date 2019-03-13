#ifndef REACTIVETRANSPORTGPU_ABSTRACT_REACTIVE_TRANSPORT_SOLVER_H
#define REACTIVETRANSPORTGPU_ABSTRACT_REACTIVE_TRANSPORT_SOLVER_H

#include <matplotlibcpp.h>

#include "profiler/profiler.hpp"

#include "common.h"
#include "misc/simple_timer.h"
#include "equilibrium_solver/equilibrium_solver.h"
#include "chemistry.h"
#include "transport.h"
#include "misc/string_format.h"

namespace plt = matplotlibcpp;

namespace reactive_transport_solver {

using namespace common;

using equilibrium_solver::EquilibriumStateSOA;
using chemistry::ThermodynamicProperties;
using chemistry::component_amounts;
using chemistry::elementAmounts;
using chemistry::elementAmountsAqueous;
using equilibrium_solver::EquilibriumState;
using equilibrium_solver::MinimizerOptions;

struct ReactiveTransportSolverConf {
    size_t ncells=1000;

    numeric_t D  = 0;      // in units of m2/s
    // D  = 1.0e-9      // in units of m2/s
    // v  = 1.0/day     // in units of m/s
    numeric_t v  = 1.0/day;     // in units of m/s

    //numeric_t dx = 1.0/ncells;     // in units of m
    // dt = 1*hour   // in units of s
    numeric_t dt = 60*minute;   // in units of s
    // dt = dx**2/D * 0.1  // in units of s

    numeric_t T = 60.0 + 273.15;  // 60 degC in K
    numeric_t P = 100 * 1e5;      // 100 bar in Pa

    numeric_t substances_bc[4] = {1.0, 0.05, 0.0, 0.0};
    numeric_t substances_ic[4] = {1.0, 0.0, 0.7, 0.1};

    MinimizerOptions minimizer_options;

    bool plot = true;

    numeric_t dx() const {
        return 1.0/ncells;
    }

    ThermodynamicProperties thermodynamic_properties() const {
        return ThermodynamicProperties(T, P);
    }
};

struct ReactiveTransportSolverAbstract {
    using bs_t = Eigen::Matrix<numeric_t, Eigen::Dynamic, common::num_components, Eigen::RowMajor>;

    const ReactiveTransportSolverConf conf;

    numeric_t t = 0;

    numeric_t progress = std::numeric_limits<double>::max();

    // components amounts in the boundary cell
    component_amounts_t b_bc;

    // component amounts in the last step
    bs_t b0;
    // component amounts in the current step
    bs_t b;
    // component amounts of components in fluid phase
    bs_t bfluid;
    // component amounts of components in fluid phase
    bs_t bsolid;

    // number of iterations per cell in current step
    std::vector<size_t> iterations;

    // number of iterations in all cells per step
    std::vector<size_t> iterations_step;

    // throughput in each step in number of iterations per second
    std::vector<double> runtime_step;

    double total_runtime = 0;
    double total_runtime_transport = 0;
    double total_runtime_equilibration = 0;

    // total number of iterations in all steps
    size_t total_iterations;

    // equilibrium states of all cells
    EquilibriumStateSOA states;

    // coordinates of the cells
    Vector<numeric_t, Eigen::Dynamic> x;

    ReactiveTransportSolverAbstract() : ReactiveTransportSolverAbstract(ReactiveTransportSolverConf()) {}

    virtual ~ReactiveTransportSolverAbstract() {}

    ReactiveTransportSolverAbstract(ReactiveTransportSolverConf conf_) :
            conf(conf_),
            b0(conf.ncells, num_components),
            b(conf.ncells, num_components),
            bfluid(conf.ncells, num_components),
            bsolid(conf.ncells, num_components),
            iterations(conf.ncells, 0),
            states(conf.ncells),
            x(conf.ncells)
    {
        // initialize cell coordinates
        x.setLinSpaced(conf.ncells, 0., 1.);

        // initialize cells
        b_bc = component_amounts(conf.substances_bc[0], conf.substances_bc[1], conf.substances_bc[2], conf.substances_bc[3]);
        auto b_ic = component_amounts(conf.substances_ic[0], conf.substances_ic[1], conf.substances_ic[2], conf.substances_ic[3]);

        EquilibriumState state_bc;
        EquilibriumState state_ic;

        equilibrate(conf.thermodynamic_properties(), b_bc, state_bc, conf.minimizer_options, true);
        auto min_res_init = equilibrate(conf.thermodynamic_properties(), b_ic, state_ic, conf.minimizer_options, true);

        for (size_t i=0; i<conf.ncells; ++i) {
            iterations[i] = min_res_init.it;
        }

        states.init(state_ic);

        // initialize components amounts vector
        b0.setZero();
        b.setZero();;
        bfluid.setZero();
        bsolid.setZero();
        for (size_t i=0; i<conf.ncells; ++i) {
            b.row(i) = elementAmounts(state_ic.n);
        }
    }

    void step() {
        if (conf.plot)
            plot();

        SimpleTimer timer_total;
        timer_total.tic();

        PROFILER_START("transport");
        SimpleTimer timer_transport;
        timer_transport.tic();
        // compute the components amounts in the aqueous and solid phase
        for (size_t icell=0; icell<conf.ncells; ++icell) {
            bfluid.row(icell) = elementAmountsAqueous(states[icell].n);
            // bsolid[icell] = elementAmountsMineral(states[icell].n)
            bsolid.row(icell) = formula_matrix.col(num_species-1) * states[icell].n[num_species-1];
        }

        // transport fluids
        transport_step();

        // update components amounts
        b = bsolid + bfluid;
        timer_transport.toc();
        total_runtime_transport += timer_transport.duration();
        PROFILER_STOP("transport");

        // equilibrate
        SimpleTimer timer;
        timer.tic();
        equilibration_step();
        timer.toc();
        total_runtime_equilibration += timer.duration();

        // update
        progress = (b - b0).norm();
        b0 = b;
        t += conf.dt;
        std::cout << "progress: " << progress << std::endl;

        // collect data
        size_t iterations_current_step=0;
        for (size_t i=0; i<conf.ncells; ++i) {
            iterations_current_step += iterations[i];
        }
        iterations_step.push_back(iterations_current_step);
        runtime_step.push_back(timer.duration());
        std::cout << "throughput [iterations/s]: " << iterations_current_step/timer.duration() << std::endl;

        timer_total.toc();
        total_runtime += timer_total.duration();
    }

    virtual std::string identifier() = 0;

    void save_statistics(std::string suffix) {
        std::ofstream f_it_step("../data/iterations_step_"+identifier()+"_"+suffix+".txt");
        for (size_t iter : iterations_step) {
            f_it_step << iter << std::endl;
        }
        f_it_step.close();

        std::ofstream f_runtime("../data/runtime_"+identifier()+"_"+suffix+".txt");
        for (double rt : runtime_step) {
            f_runtime << rt << std::endl;
        }
        f_runtime.close();

        std::ofstream f_runtime_total("../data/total_runtime_"+identifier()+"_"+suffix+".txt");
        f_runtime_total << total_runtime;
        f_runtime_total.close();

        std::ofstream f_runtime_transport_total("../data/total_runtime_transport_"+identifier()+"_"+suffix+".txt");
        f_runtime_transport_total << total_runtime_transport;
        f_runtime_transport_total.close();

        std::ofstream f_runtime_equilibration_total("../data/total_runtime_equilibration_"+identifier()+"_"+suffix+".txt");
        f_runtime_equilibration_total << total_runtime_equilibration;
        f_runtime_equilibration_total.close();
    }

    /*numeric_t average_throughput() {
        numeric_t avg_throughput;
        for (size_t i=0; i<throughput.size(); ++i) {
            avg_throughput += throughput[i];
        }
        avg_throughput/=throughput.size();
        return avg_throughput;
    }*/

    void transport_step() {
        for (size_t j=0; j<num_components; ++j) {
            // todo: check memory layout
            Vector<numeric_t, Eigen::Dynamic> tmp = bfluid.col(j);
            transport(tmp, conf.dt, conf.dx(), conf.v, conf.D, b_bc[j]);
            bfluid.col(j) = tmp;
        }
    }

    virtual void equilibration_step() = 0;

    void plot() {
        plt::clf();

        // pH = [-log10(state.a[iH]) for state in states]
        std::vector<numeric_t> nCa(conf.ncells);
        std::vector<numeric_t> nCO2(conf.ncells);
        std::vector<numeric_t> nCaCO3(conf.ncells);
        for (size_t i=0; i<conf.ncells; ++i) {
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
    }
};

}
#endif //REACTIVETRANSPORTGPU_ABSTRACT_REACTIVE_TRANSPORT_SOLVER_H
