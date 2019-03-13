#ifndef REACTIVETRANSPORT_GIBBS_ENERGY_H
#define REACTIVETRANSPORT_GIBBS_ENERGY_H

#include "common.h"
#ifdef __CUDA_ARCH__
#include "common_cuda.h"
#endif
#include "newton_1d.h"

namespace chemistry {

// make variables on the device visible
#ifdef __CUDA_ARCH__
    using namespace common_device;
#else
    using namespace common;
#endif

// Auxiliary funtion that calculates the vector b of component amounts for
// a given recipe involving H2O, CO2, NaCl, and CaCO3.
common::component_amounts_t component_amounts(numeric_t kgH2O, numeric_t molCO2, numeric_t molNaCl, numeric_t molCaCO3);

// The Drummond model for ln activity coefficient of CO2(aq).
// Parameters:
// - T is temperature in K
// - I is ionic strength in molal
// Return: the ln activity coefficient of CO2(aq)
DEVICE_DECL_SPEC HOST_DECL_SPEC
numeric_t ln_activity_coeff_co2_drummond(const numeric_t T, const numeric_t I);


// The Davies model for the ln activity coefficient of aqueous ions.
// Parameters:
// - Z is the charge of the ionic species
// - I is the ionic strength in molal
// Return: the ln of activity coefficient of the aqueous ion
DEVICE_DECL_SPEC HOST_DECL_SPEC
numeric_t ln_activity_coeff_ion_davies(const numeric_t Z, const numeric_t I);

// The Peng-Robinson model for the ln fugacity coefficient of CO2(g).temperature
// Parameters:
// - T is temperature in K
// - P is pressure in Pa.
// Return: the ln of fugacity coefficient of CO2(g)
DEVICE_DECL_SPEC HOST_DECL_SPEC
numeric_t ln_fugacity_coefficient_co2(const numeric_t T, const numeric_t P);

// Define the function that calculates the activities of all aqueous species.
// Parameters:
//   - T is temperature in units of K
//   - P is pressure in units of Pa
//   - nphase is an array with the mole amounts of aqueous species only
// Return:
//   - an array with the ln activities of all aqueous species
DEVICE_DECL_SPEC HOST_DECL_SPEC
void ln_activities_aqueous_species(const numeric_t T, const numeric_t P,
                                   const Vector<numeric_t, common::num_species>& n, const int slice[2],
                                   Vector<numeric_t, common::num_species>& ln_a);

// Define the function that calculates the activities of all gaseous species.
// Parameters:
//   - T is temperature in units of K
//   - P is pressure in units of Pa
//   - nphase is an array with the mole amounts of gaseous species only
// Return:
//   - an array with the ln activities of all gaseous species
DEVICE_DECL_SPEC HOST_DECL_SPEC
void ln_activities_gaseous_species(const numeric_t T, const numeric_t P,
                                   const Vector<numeric_t, common::num_species>& n, const int slice[2],
                                   Vector<numeric_t, common::num_species>& ln_a);

// Define the function that calculates the activities of the mineral species.
// Parameters:
//   - T is temperature in units of K
//   - P is pressure in units of Pa
//   - nphase is an array with the mole amounts of the mineral species
// Return:
//   - an array with the ln activities of mineral species
DEVICE_DECL_SPEC HOST_DECL_SPEC
void ln_activities_mineral_species(const numeric_t T, const numeric_t P,
                                   const Vector<numeric_t, common::num_species>& nphase,
                                   const int slice[2], Vector<numeric_t, common::num_species>& ln_a);

// todo: use template for nphase (is a block of a larger vector)
DEVICE_DECL_SPEC HOST_DECL_SPEC
Eigen::Matrix<numeric_t, common::num_species_aqueous, common::num_species_aqueous> ln_activities_aqueous_species_ddn(
        const numeric_t T, const numeric_t P, const Vector<numeric_t, common::num_species_aqueous> nphase);

// Define the function that calculates the activities of all species.
// Parameters:
//   - T is temperature in units of K
//   - P is pressure in units of Pa
//   - n is an array with the mole amounts of all species in the chemical system
// Return:
//   - an array with the ln activities of all species in the chemical system
DEVICE_DECL_SPEC HOST_DECL_SPEC
Vector<numeric_t, common::num_species> ln_activities(const numeric_t T, const numeric_t P, const Vector<numeric_t, common::num_species>& n);

// Define the function that calculates the partial molar derivatives of the
// ln activities of all species
DEVICE_DECL_SPEC HOST_DECL_SPEC
common::hessian_t ln_activities_ddn(const numeric_t T, const numeric_t P, const Vector<numeric_t, common::num_species>& n);

// Define the function that calculates the concentrations of the species.
// For aqueous solutes, concentrations are molalities.
// For aqueous solvent H2O(l), mole fraction.
// For gaseous species, partial pressure.
// For mineral species, mole fractions.
DEVICE_DECL_SPEC HOST_DECL_SPEC
Vector<numeric_t, common::num_species> concentrations(numeric_t T, numeric_t P, const Vector<numeric_t, common::num_species>& n);

// Define the function that calculates the masses of all species.
DEVICE_DECL_SPEC HOST_DECL_SPEC
Vector<numeric_t, common::num_species> masses(const Vector<numeric_t, common::num_species>& n);

// Define the function that calculates the amounts of elements
DEVICE_DECL_SPEC HOST_DECL_SPEC
Vector<numeric_t, common::num_components> elementAmounts(const Vector<numeric_t, common::num_species>& n);

// Return the amounts of elements in the aqueous phase
DEVICE_DECL_SPEC HOST_DECL_SPEC
Vector<numeric_t, common::num_components> elementAmountsAqueous(const Vector<numeric_t, common::num_species>& n);

// Return the amounts of elements in the gaseous phase
DEVICE_DECL_SPEC HOST_DECL_SPEC
Vector<numeric_t, common::num_components> elementAmountsGaseous(const Vector<numeric_t, common::num_species>& n);

// Return the amounts of elements in the mineral phase
DEVICE_DECL_SPEC HOST_DECL_SPEC
Vector<numeric_t, common::num_components> elementAmountsMineral(const Vector<numeric_t, common::num_species>& n);

// Define the function that calculates the standard chemical potentials of all
// species at given temperature and pressure.
// Parameters:
//   - T is temperature in units of K
//   - P is pressure in units of Pa
// Return:
//   - an array with the standard chemical potentials of all species (in J/mol)
DEVICE_DECL_SPEC HOST_DECL_SPEC
Vector<numeric_t, common::num_species> standard_chemical_potentials(const numeric_t T, const numeric_t P);

struct ThermodynamicProperties {
    const numeric_t T; // temperature
    const numeric_t P; // pressure
    const Vector<numeric_t, common::num_species> u0; // standard chemical potentials
    const numeric_t ln_fugacity_c02;

    DEVICE_DECL_SPEC HOST_DECL_SPEC
    inline ThermodynamicProperties(numeric_t T_, numeric_t P_) :
            T(T_),
            P(P_),
            u0(chemistry::standard_chemical_potentials(T, P)),
            ln_fugacity_c02(ln_fugacity_coefficient_co2(T, P)) {}

    DEVICE_DECL_SPEC HOST_DECL_SPEC
    inline const Vector<numeric_t, common::num_species>& standard_chemical_potentials() const {
        return u0;
    };
};

// Define the function that calculates the chemical potentials of all species
// at given temperature, pressure, and mole amounts of species.
// Parameters:
//   - T is temperature in units of K
//   - P is pressure in units of Pa
//   - n is an array with the mole amounts of the species
// Return:
//   - an array with the chemical potentials of all species (in J/mol)
DEVICE_DECL_SPEC HOST_DECL_SPEC
Vector<numeric_t, common::num_species> chemical_potentials(const ThermodynamicProperties& thermo_props, const Vector<numeric_t, common::num_species>& n);

// Define the function that calculates the partial molar derivatives of the
// chemical potentials of the species.
DEVICE_DECL_SPEC HOST_DECL_SPEC
common::hessian_t chemical_potentials_ddn(const ThermodynamicProperties& thermo_props, const Vector<numeric_t, common::num_species>& n);

struct ObjectiveResult {
    numeric_t f;
    common::gradient_t g;
    common::hessian_t H;

    DEVICE_DECL_SPEC HOST_DECL_SPEC ObjectiveResult() : f(0) {
        g.setConstant(0);
        H.setConstant(0);
    }
};

// Define the function that calculates:
//  - G, the Gibbs energy of the system;
//  - u, the gradient of the Gibbs energy, or chemical potentials of the species
//  - H, the Hessian of the Gibbs energy, or partial molar derivatives of activities
// These quantities are normalized by RT for numerical reasons.
DEVICE_DECL_SPEC HOST_DECL_SPEC
ObjectiveResult gibbs_energy(const ThermodynamicProperties& thermo_props, const Vector<numeric_t, common::num_species>& n);

DEVICE_DECL_SPEC HOST_DECL_SPEC
ObjectiveResult gibbs_energy_optimized(const ThermodynamicProperties& thermo_props, const Vector<numeric_t, common::num_species>& n);

// Define the function that calculates the Gibbs energy function for a chemical
// system in which all species are assumed as pure phases. This is needed for
// calculation of an initial guess for the species amounts that satisfies the
// mass conservation conditions.
DEVICE_DECL_SPEC HOST_DECL_SPEC
ObjectiveResult gibbs_energy_pure_phases(const ThermodynamicProperties& thermo_props, const Vector<numeric_t, common::num_species>& n);


template <typename HESSIAN_T>
DEVICE_DECL_SPEC HOST_DECL_SPEC
void gibbs_energy_opt_inplace_hessian(const ThermodynamicProperties& thermo_props,
                                         const Vector<numeric_t, common::num_species>& n,
                                         HESSIAN_T&& H) {
    /*
     * calculate Hessian
     */
    // The molar amount of H2O(l)
    numeric_t nH2O = n[iH2O];
    numeric_t nH2O_inv = 1/n[iH2O];

    // The mole fraction of H2O(l)
    numeric_t xH2O = nH2O/n.segment(slice_aqueous[0], slice_aqueous[1]-slice_aqueous[0]).sum();

    for (int i=slice_aqueous[0]; i<slice_aqueous[1]; ++i) {
        H(i, i) = 1/n[i];
    }
    for (int i=slice_aqueous[0]; i<slice_aqueous[1]; ++i) {
        H(i, iH2O) = -nH2O_inv;
    }
    for (int i=slice_aqueous[0]; i<slice_aqueous[1]; ++i) {
        H(iH2O, i) = -nH2O_inv * xH2O/(xH2O - 1.0);
    }
    for (int i=slice_aqueous[0]; i<slice_aqueous[1]; ++i) {
        H(iH2O, iH2O) = -nH2O_inv;
    }
}

template <typename GRADIENT_T>
DEVICE_DECL_SPEC HOST_DECL_SPEC
void gibbs_energy_opt_inplace_gradient(const ThermodynamicProperties& thermo_props,
                                         const Vector<numeric_t, common::num_species>& n,
                                         GRADIENT_T&& g) {
    const numeric_t T = thermo_props.T;
    const numeric_t P = thermo_props.P;
    const numeric_t RT = R*T;

    /*
     * calculate gradient
     */
    auto& ln_a = g; // store the activities directly in the gradient
    ln_a.setConstant(0);
    {
        numeric_t I = 0.;
        numeric_t nphase_sum = 0.;

        for (size_t i = slice_aqueous[0]; i<slice_aqueous[1]; ++i) {
            nphase_sum += n[i];
            double m = 55.508 * n[i] / n[iH2O];
            ln_a[i] += std::log(m);

            numeric_t Zi = charges[i];
            I += m*Zi*Zi;
        }

        I *= 0.5;

        // Calculate the mole fraction of water species, H2O(l)
        numeric_t xH2O = n[iH2O] / nphase_sum;

        ///////////////////////////////////////////////////////////////////////////
        // Calculate the ln activity coefficient of CO2(aq)
        // ---- USE DRUMMOND (1981) MODEL ----
        if (!useideal) {
            constexpr numeric_t c1 = -1.0312;
            constexpr numeric_t c2 = 1.2806e-3;
            constexpr numeric_t c3 = 255.9;
            constexpr numeric_t c4 = 0.4445;
            constexpr numeric_t c5 = -1.606e-3;
            ln_a[iCO2] += (c1 + c2*T + c3/T)*I - (c4 + c5*T)*I/(1 + I);
        }
        ///////////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////////
        // Calculate the ln activity coefficient of each charged species
        // ---- USE DAVIES MODEL FOR EACH CHARGED SPECIES ----
        for (int i=0; i<common::num_charged_species(); ++i) {
            constexpr numeric_t Agamma = 0.5095;
            const numeric_t sqrtI = sqrt(I);

            ln_a[icharged[i]] += 2.30258 * (-Agamma * charges[i]*charges[i] * (sqrtI / (1.0 + sqrtI) - 0.3 * I));
        }
        ///////////////////////////////////////////////////////////////////////////

        // Calculate the ln activity of the solvent species H2O(l)
        ln_a[iH2O] = -(1 - xH2O)/xH2O;
    }
    {
        // Calculate the array of mole fractions of the gaseous species
        numeric_t nphase_sum=0;
        for (size_t i=slice_gaseous[0]; i<slice_gaseous[1]; ++i) {
            nphase_sum += n[i];
        }
        for (size_t i=slice_gaseous[0]; i<slice_gaseous[1]; ++i) {
            ln_a[i] += std::log(n[i]/nphase_sum);
        }

        ///////////////////////////////////////////////////////////////////////////
        // Calculate the ln fugacity coefficient of the only gaseous species, CO2(g)
        // ---- USE PENG-ROBINSON (1976) MODEL ----
        ln_a[slice_gaseous[0]] += useideal ? 0.0 : ln_fugacity_coefficient_co2(T, P);
        ///////////////////////////////////////////////////////////////////////////

        // Calculate the ln activities of the gaseous species
        ln_a.segment(slice_gaseous[0], slice_gaseous[1]-slice_gaseous[0]) += std::log(P/Pref)*Vector<numeric_t, common::num_species_gaseous>::Ones();
    }

    g += thermo_props.standard_chemical_potentials()/RT;
}

#ifdef USE_CUDA
void gibbs_energy_batch_gpu(ThermodynamicProperties thermo_props,
                            int nevals,
                            Vector<numeric_t, common::num_species>* xs,
                            ObjectiveResult* results);
#endif

}

#endif //REACTIVETRANSPORT_GIBBS_ENERGY_H
