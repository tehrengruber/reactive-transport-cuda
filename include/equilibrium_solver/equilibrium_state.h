#ifndef REACTIVETRANSPORTGPU_EQUILIBRIUM_STATE_H
#define REACTIVETRANSPORTGPU_EQUILIBRIUM_STATE_H

#include <iomanip>
#include <vector>
#include "common.h"
#include "cuda/managed_allocator.hpp"

namespace equilibrium_solver {

struct EquilibriumState {
    numeric_t T;                      // temperature in K
    numeric_t P;                      // pressure in Pa
    Vector<numeric_t, common::num_species>& n; // amounts of species in mol
    Vector<numeric_t, common::num_species> m; // masses of species in g
    Vector<numeric_t, common::num_species> c; // concentrations of the species
    Vector<numeric_t, common::num_species> a; // activities of the species
    Vector<numeric_t, common::num_species> g; // activity coefficients of the species
    Vector<numeric_t, common::num_species> x; // alias to amounts of species in mol
    Vector<numeric_t, common::num_components> y; // Lagrange multipliers y in J/mol
    Vector<numeric_t, common::num_species> z; // Lagrange multipliers z in J/mol

    EquilibriumState() : n(x) {}
#ifndef __CUDA_ARCH__
    std::ostream& operator<<(std::ostream& stream) {
        std::stringstream bar_ss;
        for (size_t i=0; i<150; ++i) { // bar
            bar_ss << "=";
        }
        std::stringstream thinbar_ss;
        for (size_t i=0; i<150; ++i) { // bar
            thinbar_ss << "-";
        }
        std::string bar = bar_ss.str();
        std::string thinbar = thinbar_ss.str();
        stream << bar;
        stream << std::setw(25) << "Temperature [K]"
               << std::setw(25) << "Temperature [C]"
               << std::setw(25) << "Pressure [Pa]"
               << std::setw(25) << "Pressure [bar]";
        stream << thinbar;
        stream << std::setw(25) << T
               << std::setw(25) << T - 273.15
               << std::setw(25) << P
               << std::setw(25) << P * 1e-5;
        stream << bar;
        stream << std::setw(25) << "Species"
               << std::setw(25) << "Amount [mol]"
               << std::setw(25) << "Mass [g]"
               << std::setw(25) << "Concentration"
               << std::setw(25) << "Activity"
               << std::setw(25) << "Activity Coefficient";
        stream << thinbar;
        for (size_t i=0; i<common::num_species; ++i) {
            stream << std::setw(25) << common::species[i]
                   << std::setw(25) << n[i]
                   << std::setw(25) << m[i]
                   << std::setw(25) << c[i]
                   << std::setw(25) << a[i]
                   << std::setw(25) << g[i];
            stream << bar;
        }
        return stream;
    }
#endif
};

struct EquilibriumStateRef {
    Vector<numeric_t, common::num_species>& n; // amounts of species in mol
    Vector<numeric_t, common::num_species>& m; // masses of species in g
    Vector<numeric_t, common::num_species>& c; // concentrations of the species
    Vector<numeric_t, common::num_species>& a; // activities of the species
    Vector<numeric_t, common::num_species>& g; // activity coefficients of the species
    Vector<numeric_t, common::num_species>& x; // alias to amounts of species in mol
    Vector<numeric_t, common::num_components>& y; // Lagrange multipliers y in J/mol
    Vector<numeric_t, common::num_species>& z; // Lagrange multipliers z in J/mol

    template <typename T>
    EquilibriumStateRef(T& states, const size_t i) :
            n(states.n[i]),
            m(states.m[i]),
            c(states.c[i]),
            a(states.a[i]),
            g(states.g[i]),
            x(states.x[i]),
            y(states.y[i]),
            z(states.z[i]) {}
};

struct EquilibriumStateSOA {
    size_t size_;

    using vec_species = Vector<numeric_t, common::num_species>;
    using vec_components = Vector<numeric_t, common::num_components>;

    #ifdef USE_CUDA
    std::vector<vec_species, managed_allocator<vec_species>> n; // amounts of species in mol
    std::vector<vec_species> m; // masses of species in g
    std::vector<vec_species> c; // concentrations of the species
    std::vector<vec_species> a; // activities of the species
    std::vector<vec_species> g; // activity coefficients of the species
    std::vector<vec_species, managed_allocator<vec_species>>& x; // alias to amounts of species in mol
    std::vector<vec_components, managed_allocator<vec_components>> y; // Lagrange multipliers y in J/mol
    std::vector<vec_species, managed_allocator<vec_species>> z; // Lagrange multipliers z in J/mol
    #else
    std::vector<vec_species> n; // amounts of species in mol
    std::vector<vec_species> m; // masses of species in g
    std::vector<vec_species> c; // concentrations of the species
    std::vector<vec_species> a; // activities of the species
    std::vector<vec_species> g; // activity coefficients of the species
    std::vector<vec_species>& x; // alias to amounts of species in mol
    std::vector<vec_components> y; // Lagrange multipliers y in J/mol
    std::vector<vec_species> z; // Lagrange multipliers z in J/mol
    #endif

    EquilibriumStateSOA(size_t size, EquilibriumState& ic) :
            size_(size),
            n(size, ic.n),
            m(size, ic.m),
            c(size, ic.c),
            a(size, ic.a),
            g(size, ic.g),
            x(n),
            y(size, ic.y),
            z(size, ic.z) {}

    EquilibriumStateSOA(size_t size) : size_(size), x(n) {
        n.reserve(size);
        m.reserve(size);
        c.reserve(size);
        a.reserve(size);
        g.reserve(size);
        y.reserve(size);
        z.reserve(size);
    }

    void init(const EquilibriumState& es) {
        n.resize(size(), es.n);
        m.resize(size(), es.m);
        c.resize(size(), es.c);
        a.resize(size(), es.a);
        g.resize(size(), es.g);
        y.resize(size(), es.y);
        z.resize(size(), es.z);
    }

    EquilibriumStateRef operator[](const size_t i) {
        return EquilibriumStateRef(*this, i);
    }

    size_t size() {
        return size_;
    }
};

}

#endif //REACTIVETRANSPORTGPU_EQUILIBRIUM_STATE_H
