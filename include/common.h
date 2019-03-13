#ifndef REAKTIVETRANSPORTGPU_SETUP_H
#define REAKTIVETRANSPORTGPU_SETUP_H

#include <map>
#include <vector>

#include <iostream>

#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Dense>

//#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#ifdef __CUDA_ARCH__
#define DEVICE_DECL_SPEC __device__
#define HOST_DECL_SPEC __host__
#else
#define DEVICE_DECL_SPEC
#define HOST_DECL_SPEC
#endif

#ifdef __CUDA_ARCH__
#define HOST_VER_ONLY(expr)
#else
#define HOST_VER_ONLY(expr) expr
#endif

using namespace Eigen;

template <typename T, int N>
using Vector = Eigen::Matrix<T, N, 1>;

using numeric_t = double;

template <typename T>
static void escape (T&& p) {
    asm volatile("" :  : "g"(p));
}

static void clobber()
{
    asm volatile("" : : : "memory");
}

static void* dummy = nullptr;

#define CACHE_SIZE 8192*1024
static void clear_cpu_cache() {
    if (dummy==nullptr)
        dummy = new char[CACHE_SIZE];
    memset (dummy, 0, CACHE_SIZE);
    escape(dummy);
    //delete[] dummy;
}

#ifdef USE_MULTIPLE_THREADS
#define ST_VER_ONLY(expr)
#else
#define ST_VER_ONLY(expr) expr
#endif

// Auxiliary time related constants
const numeric_t second = 1.0;
const numeric_t minute = 60.0;
const numeric_t hour = 60 * minute;
const numeric_t day = 24 * hour;
const numeric_t year = 365 * day;

template <int N>
std::vector<numeric_t> stdvec_from_eigvec(const Vector<numeric_t, N> v) {
    return std::vector<numeric_t>(&v[0], &v[0]+v.size());
}

using keyword_map_t = std::map<std::string, std::string>;

static inline keyword_map_t make_map(std::initializer_list<std::string> list) {
    using it_t = std::initializer_list<std::string>::iterator;
    assert(list.size()%2==0);

    keyword_map_t map;
    for (it_t it=list.begin(); it < list.end(); it+=2) {
        map[*it] = *(it+1);
    }

    return map;
};

namespace common {

// The universal gas constant in units of J/(mol*K)
static constexpr numeric_t R = 8.3144598;

// The reference pressure P(ref) (in units of Pa)
static constexpr numeric_t Pref = 1.0e5;

// Create a list of component names
static const char* components[6] = { "H", "O", "C", "Na", "Cl", "Ca" };

// Create a list of species names
static const char* species[11] = {
        "H2O(l)",
        "H+(aq)",
        "OH-(aq)",
        "HCO3-(aq)",
        "CO3--(aq)",
        "CO2(aq)",
        "Na+(aq)",
        "Cl-(aq)",
        "Ca++(aq)",
        "CO2(g)",
        "CaCO3(s,calcite)"
};

// The number of components and species
static constexpr int num_components = 6; // components.size()
static constexpr int num_species = 11; // species.size()

// The number of aqueous, gaseous, and mineral species
static constexpr int num_species_gaseous = 1;
static constexpr int num_species_mineral = 1;
static constexpr int num_species_aqueous = 9;

static constexpr int slice_aqueous[2] = {0, num_species_aqueous};
static constexpr int slice_gaseous[2] = {num_species_aqueous, num_species_aqueous + num_species_gaseous};
static constexpr int slice_mineral[2] = {num_species_aqueous + num_species_gaseous, num_species};

// Create a list with the charges of the aqueous species
static constexpr int charges[num_species_aqueous] = { 0, 1, -1, -1, -2, 0, 1, -1, 2 };

// A function that validates that the given index corresponds to the given species
#if defined(__CUDA_ARCH__) || defined(NODEBUG)
#define SPECIES_INDEX_VALIDATE(s, index) index
#else
inline int species_index_validate(const char* s, const int index) {
    assert(std::find(species, species+num_species, std::string(s))-species == index);
    return index;
}
#define SPECIES_INDEX_VALIDATE(s, index) common::species_index_validate(s, index)
#endif

// Store the index of H2O(l)
static const int iH2O = SPECIES_INDEX_VALIDATE("H2O(l)", 0);

// Store the index of H+(aq)
static const int iH = SPECIES_INDEX_VALIDATE("H+(aq)", 1);

// Store the index of CO2(aq)
static const int iCO2 = SPECIES_INDEX_VALIDATE("CO2(aq)", 5);

// Store the index of Ca++(aq)
static const int iCa = SPECIES_INDEX_VALIDATE("Ca++(aq)", 8);

// Store the index of CaCO3(s,calcite)
static const int iCaCO3 = SPECIES_INDEX_VALIDATE("CaCO3(s,calcite)", 10);

DEVICE_DECL_SPEC HOST_DECL_SPEC
constexpr int num_charged_species() {
    return 7;
}

// Create a list with the indices of the charged species
static int icharged[num_charged_species()] = {1, 2, 3, 4, 6, 7, 8};

// Construct the formula matrix of the chemical system
using formula_matrix_t = Eigen::Matrix<numeric_t, num_components, num_species>;
using formula_matrix_trans_t = Eigen::Matrix<numeric_t, num_species, num_components>;

static formula_matrix_t formula_matrix;

}

#include "chem_pot_table.h"

namespace common {

static ChemPotTable u0_table;

// Create an array with the molar masses of the components
static Vector<numeric_t, num_components> components_molar_masses;

// Create an array with the molar masses of the species
static Vector<numeric_t, num_species> species_molar_masses;

// A global variable to allow one to choose if ideal activity models are used
static constexpr bool useideal = false;

// A global variable to allow one to choose only reference state standard
// chemical potentials are used, without interpolation
static constexpr bool userefpotentials = false;

using gradient_t = Vector<numeric_t, num_species>;
using hessian_t = Eigen::Matrix<numeric_t, num_species, num_species>;

using component_amounts_t = Vector<numeric_t, num_components>;

namespace detail {

    static void* initialize() {
        int c=0;
        for (int i=0; i<num_species_aqueous; ++i) {
            if (charges[i] != 0) {
                icharged[c] = i;
                ++c;
            }
        }
        assert(c==num_charged_species());

        // initialize formula matrix
        formula_matrix << 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, // H
                1, 0, 1, 3, 3, 2, 0, 0, 0, 2, 3, // O
                0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, // C
                0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, // Na
                0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, // Cl
                0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1; // Ca
        //0, 1,-1,-1,-2, 0, 1,-1, 2, 0, 0; // Z

        // initialize array with the molar masses of the species
        components_molar_masses << 1.0079,   // H
                15.9994,  // O
                12.0107,  // C
                22.9898,  // Na
                35.453,   // Cl
                40.078;  // Ca

        species_molar_masses = formula_matrix.transpose() * components_molar_masses;

        return nullptr;
    }

static void* _dummy = initialize();
} // end ns detail

} // end ns common

#endif //REAKTIVETRANSPORTGPU_SETUP_H
