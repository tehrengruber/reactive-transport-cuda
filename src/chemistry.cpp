#include "common_cuda.h"
#include "newton_1d.h"
#include "chemistry.h"

#ifdef USE_CUDA
#include "kernel/gibbs_energy_kernel_simple.cu"
#endif

namespace chemistry {

// make variables on the device visible
#ifdef __CUDA_ARCH__
using namespace common_device;
#else
using namespace common;
#endif

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

// The Drummond model for ln activity coefficient of CO2(aq).
// Parameters:
// - T is temperature in K
// - I is ionic strength in molal
// Return: the ln activity coefficient of CO2(aq)
DEVICE_DECL_SPEC HOST_DECL_SPEC
numeric_t ln_activity_coeff_co2_drummond(const numeric_t T, const numeric_t I) {
    const numeric_t c1 = -1.0312;
    const numeric_t c2 = 1.2806e-3;
    const numeric_t c3 = 255.9;
    const numeric_t c4 = 0.4445;
    const numeric_t c5 = -1.606e-3;
    return (c1 + c2*T + c3/T)*I - (c4 + c5*T)*I/(1 + I);
}


// The Davies model for the ln activity coefficient of aqueous ions.
// Parameters:
// - Z is the charge of the ionic species
// - I is the ionic strength in molal
// Return: the ln of activity coefficient of the aqueous ion
DEVICE_DECL_SPEC HOST_DECL_SPEC
numeric_t ln_activity_coeff_ion_davies(const numeric_t Z, const numeric_t I) {
    const numeric_t Agamma = 0.5095;
    const numeric_t sqrtI = sqrt(I);

    return 2.30258 * (-Agamma * Z*Z * (sqrtI / (1.0 + sqrtI) - 0.3 * I));
}

// The Peng-Robinson model for the ln fugacity coefficient of CO2(g).temperature
// Parameters:
// - T is temperature in K
// - P is pressure in Pa.
// Return: the ln of fugacity coefficient of CO2(g)
DEVICE_DECL_SPEC HOST_DECL_SPEC
numeric_t ln_fugacity_coefficient_co2(const numeric_t T, const numeric_t P) {
    // The critical temperature and pressure of CO2
    numeric_t Tc = 304.2;    // CO2 critical temperature in K
    numeric_t Pc = 73.83e5;  // CO2 critical pressure in Pa
    // The acentric factor of CO2
    numeric_t omega = 0.224;
    // Create variables for the parameters of Peng-Robinson EOS
    numeric_t epsilon = 1 - sqrt(2.0);
    numeric_t sigma = 1 + sqrt(2.0);
    numeric_t Omega = 0.07780;
    numeric_t Psi = 0.45724;
    // The reduced temperature, Tr, and reduced pressure, Pr
    numeric_t Tr = T / Tc;
    numeric_t Pr = P / Pc;
    // The evaluation of the alpha(Tr, omega) function
    numeric_t alpha = 1 + (0.37464 + 1.54226 * omega - 0.26992 * omega*omega) * (1 - sqrt(Tr));
    alpha*=alpha;
    // The beta and q contants
    numeric_t beta = Omega * Pr / Tr;
    numeric_t q = Psi / Omega * alpha / Tr;
    // Define the function f that represents the nonlinear equation f(x) = 0
    auto f = [&](numeric_t Z) {
        return (1 + beta - q * beta * (Z - beta) / ((Z + epsilon * beta) * (Z + sigma * beta))) - Z;
    };
    // Define the first order derivative of function f'(x)
    auto fprime = [&](numeric_t Z) {
        numeric_t aux = (Z + epsilon * beta) * (Z + sigma * beta);
        return -q * beta / aux * (1.0 - (Z - beta) * (2 * Z + (epsilon + sigma) * beta) / aux) - 1;
    };
    // Calculate the compressibility factor of CO2 at (T, P)
    numeric_t Z0 = 1.0; // the initial guess for the compressibility factor
    numeric_t Z = newton(f, fprime, Z0); // use newton function to perform the calculation of Z
    // Calcute theta
    numeric_t theta = 1.0 / (sigma - epsilon) * log((Z + sigma * beta) / (Z + epsilon * beta));
    // Calculate ln_phiCO2
    numeric_t ln_phiCO2 = Z - 1 - log(Z - beta) - q * theta;

    return ln_phiCO2;
}

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
                                   Vector<numeric_t, common::num_species>& ln_a) {
    // TODO: the indexing appears to be wrong in the original script and for
    //  reproducability we stick to this error for now.
    // TODO: check that this auto doesn't lead to errors
    auto& nphase = n.segment(slice[0], slice[1]-slice[0]);

    // Calculate the molalities of the aqueous species and their natural log
    Vector<numeric_t, common::num_species_aqueous> m = 55.508 * nphase / nphase[iH2O];

    // Calculate the ionic strength of the aqueous phase
    numeric_t I = 0.;
    for (int i=0; i<num_species_aqueous; ++i) {
        auto Zi = charges[i];
        I += m[i]*Zi*Zi;
    }
    I *= 0.5;

    // Calculate the mole fraction of water species, H2O(l)
    numeric_t xH2O = nphase[iH2O] / nphase.sum();

    // Create an array to store the ln activity coeffs of the aqueous species
    Vector<numeric_t, common::num_species_aqueous> ln_g;
    ln_g.setConstant(0);

    ///////////////////////////////////////////////////////////////////////////
    // Calculate the ln activity coefficient of CO2(aq)
    // ---- USE DRUMMOND (1981) MODEL ----
    if (useideal)
        ln_g[iCO2] = 0.0;
    else
        ln_g[iCO2] = ln_activity_coeff_co2_drummond(T, I);
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // Calculate the ln activity coefficient of each charged species
    // ---- USE DAVIES MODEL FOR EACH CHARGED SPECIES ----
    for (int i=0; i<common::num_charged_species(); ++i) {
        ln_g[icharged[i]] = useideal ? 0.0 : ln_activity_coeff_ion_davies(charges[i], I);
    }
    ///////////////////////////////////////////////////////////////////////////

    // Calculate the natural log of the molalities of the aqueous species
    auto& ln_m = m.array().log().matrix();

    // Calculate the ln activities of the solute species
    ln_a.segment(slice[0], slice[1]-slice[0]) = ln_g + ln_m;

    // Calculate the ln activity of the solvent species H2O(l)
    ln_a[iH2O] = -(1 - xH2O)/xH2O;
};

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
                                   Vector<numeric_t, common::num_species>& ln_a) {

    // TODO: check that using auto doesn't lead to problems here
    auto nphase = n.segment(slice[0], slice[1]-slice[0]);

    // Calculate the array of mole fractions of the gaseous species
    Vector<numeric_t, common::num_species_gaseous> x = nphase/nphase.sum();

    // Create an array to store the ln fugacity coeffs of the gaseous species
    Vector<numeric_t, common::num_species_gaseous> ln_phi;
    ln_phi.setZero();

    ///////////////////////////////////////////////////////////////////////////
    // Calculate the ln fugacity coefficient of the only gaseous species, CO2(g)
    // ---- USE PENG-ROBINSON (1976) MODEL ----
    ln_phi[0] = useideal ? 0.0 : ln_fugacity_coefficient_co2(T, P);
    ///////////////////////////////////////////////////////////////////////////

    // Calculate the array of ln mole fractions of the gaseous species
    auto ln_x = x.array().log().matrix();

    // Calculate the ln activities of the gaseous species
    ln_a.segment(slice[0], slice[1]-slice[0]) = ln_phi + ln_x + std::log(P/Pref)*Vector<numeric_t, common::num_species_gaseous>::Ones();
}

// Define the function that calculates the activities of the mineral species.
// Parameters:
//   - T is temperature in units of K
//   - P is pressure in units of Pa
//   - nphase is an array with the mole amounts of the mineral species
// Return:
//   - an array with the ln activities of mineral species
DEVICE_DECL_SPEC HOST_DECL_SPEC
void ln_activities_mineral_species(const numeric_t T, const numeric_t P,
                                   const Vector<numeric_t, common::num_species>& nphase, const int slice[2], Vector<numeric_t, common::num_species>& ln_a) {
    ln_a.segment(slice[0], slice[1]-slice[0]).setConstant(0);
}

// todo: use template for nphase (is a block of a larger vector)
DEVICE_DECL_SPEC HOST_DECL_SPEC
Eigen::Matrix<numeric_t, common::num_species_aqueous, common::num_species_aqueous> ln_activities_aqueous_species_ddn(
        const numeric_t T, const numeric_t P, const Vector<numeric_t, common::num_species_aqueous> nphase) {
    // The molar amount of H2O(l)
    numeric_t nH2O = nphase[iH2O];

    // The mole fraction of H2O(l)
    numeric_t xH2O = nH2O/nphase.sum();

    // Calculate the partial molar derivatives of the solute activities
    Eigen::Matrix<numeric_t, common::num_species_aqueous, common::num_species_aqueous> ddn;
    ddn.setConstant(0);
    ddn.diagonal() = nphase.cwiseInverse();
    ddn.col(iH2O).setConstant(-1.0/nH2O);

    // Calculate the partial molar derivatives of the solvent activity
    ddn.row(iH2O).setConstant(-1.0/nH2O * xH2O/(xH2O - 1.0));
    ddn(iH2O, iH2O) = -1.0/nH2O;

    return ddn;
}

// Define the function that calculates the activities of all species.
// Parameters:
//   - T is temperature in units of K
//   - P is pressure in units of Pa
//   - n is an array with the mole amounts of all species in the chemical system
// Return:
//   - an array with the ln activities of all species in the chemical system
DEVICE_DECL_SPEC HOST_DECL_SPEC
Vector<numeric_t, common::num_species> ln_activities(const numeric_t T, const numeric_t P, const Vector<numeric_t, common::num_species>& n) {
    // Calculate the activities of aqueous, gaseous and mineral species and
    // concatentrate them into a single array with the ln activities of all
    // species in the chemical system.
    Vector<numeric_t, common::num_species> ln_a;
    ln_a.setZero();
    ln_activities_aqueous_species(T, P, n, slice_aqueous, ln_a);
    ln_activities_gaseous_species(T, P, n, slice_gaseous, ln_a);
    ln_activities_mineral_species(T, P, n, slice_mineral, ln_a);

    return ln_a;
};

// Define the function that calculates the partial molar derivatives of the
// ln activities of all species
DEVICE_DECL_SPEC HOST_DECL_SPEC
common::hessian_t ln_activities_ddn(const numeric_t T, const numeric_t P, const Vector<numeric_t, common::num_species>& n) {
    // Create an array with the entries in n corresponding to aqueous species
    // TODO: check that this auto doesn't lead to errors
    auto n_aqueous = n.segment(slice_aqueous[0], slice_aqueous[1]-slice_aqueous[0]);

    // The matrix with partial molar derivatives of the activities
    common::hessian_t ln_a_ddn;
    ln_a_ddn.setConstant(0);

    ln_a_ddn.block(slice_aqueous[0], slice_aqueous[0],
                   slice_aqueous[1]-slice_aqueous[0], slice_aqueous[1]-slice_aqueous[0]) =
            ln_activities_aqueous_species_ddn(T, P, n_aqueous);

    return ln_a_ddn;
}

// Define the function that calculates the concentrations of the species.
// For aqueous solutes, concentrations are molalities.
// For aqueous solvent H2O(l), mole fraction.
// For gaseous species, partial pressure.
// For mineral species, mole fractions.
DEVICE_DECL_SPEC HOST_DECL_SPEC
Vector<numeric_t, common::num_species> concentrations(numeric_t T, numeric_t P, const Vector<numeric_t, common::num_species>& n) {
    Vector<numeric_t, common::num_species> c;
    c.setConstant(1);

    for (int i=slice_aqueous[0]; i<slice_aqueous[1]; ++i) {
        c[i] = 55.508 * n[i]/n[iH2O];
    }
    c[iH2O] = n[iH2O]/n.segment(slice_aqueous[0], slice_aqueous[1]-slice_aqueous[0]).sum();
    numeric_t aux = P/(Pref*n.segment(slice_gaseous[0], slice_gaseous[1]-slice_gaseous[0]).sum());
    for (int i=slice_gaseous[0]; i<slice_gaseous[1]; ++i) {
        c[i] = n[i]*aux;
    }
    c.segment(slice_mineral[0], slice_mineral[1]-slice_mineral[0]).setConstant(1);
    return c;
}


// Define the function that calculates the masses of all species.
DEVICE_DECL_SPEC HOST_DECL_SPEC
Vector<numeric_t, common::num_species> masses(const Vector<numeric_t, common::num_species>& n) {
    return (species_molar_masses.array() * n.array()).matrix();
}

// Define the function that calculates the amounts of elements
DEVICE_DECL_SPEC HOST_DECL_SPEC
Vector<numeric_t, common::num_components> elementAmounts(const Vector<numeric_t, common::num_species>& n) {
    return formula_matrix * n;
}

// Return the amounts of elements in the aqueous phase
DEVICE_DECL_SPEC HOST_DECL_SPEC
Vector<numeric_t, common::num_components> elementAmountsAqueous(const Vector<numeric_t, common::num_species>& n) {
    return formula_matrix.block(0, slice_aqueous[0], formula_matrix.rows(), slice_aqueous[1]-slice_aqueous[0])
           * n.segment(slice_aqueous[0], slice_aqueous[1]-slice_aqueous[0]);
}

// Return the amounts of elements in the gaseous phase
DEVICE_DECL_SPEC HOST_DECL_SPEC
Vector<numeric_t, common::num_components> elementAmountsGaseous(const Vector<numeric_t, common::num_species>& n) {
    return formula_matrix.block(0, slice_gaseous[0], formula_matrix.rows(), slice_gaseous[1]-slice_gaseous[0])
           * n.segment(slice_gaseous[0], slice_gaseous[1]-slice_gaseous[0]);
}

// Return the amounts of elements in the mineral phase
DEVICE_DECL_SPEC HOST_DECL_SPEC
Vector<numeric_t, common::num_components> elementAmountsMineral(const Vector<numeric_t, common::num_species>& n) {
    return formula_matrix.block(0, slice_mineral[0], formula_matrix.rows(), slice_mineral[1]-slice_mineral[0])
           * n.segment(slice_mineral[0], slice_mineral[1]-slice_mineral[0]);
}

// Find the fist index i such that values[i] <= val <= values[i+1].
// Parameters:
// - val is the given value
// - values is the list of sorted values in whic
// Example:
// >>> print index_interpolation_point(60.0, [0.0, 25.0, 50.0, 100.0, 150.0])
// 3
// >>> print index_interpolation_point(150.0, [0.0, 25.0, 50.0, 100.0, 150.0])
// 3
template <int N>
DEVICE_DECL_SPEC HOST_DECL_SPEC
int index_interpolation_point(const numeric_t val, const numeric_t (&values)[N]) {
    // TODO: strange way to code this
    for (int i=0; i<N; ++i) {
        if (values[i] > val) {
            return i-1;
        } else if (values[i] == val) {
            return i+1 < N ? i : i-1;
        }
    }
    assert(false);
    return -1;
}

// Define the interpolation function.
// Parameters:
//   - x is the x-coordinate where the interpolation is performed
//   - y is the y-coordinate where the interpolation is performed
//   - xpoints is an array with the x-coordinates where fvalues are available
//   - ypoints is an array with the y-coordinates where fvalues are available
//   - fvalues is a 2D array with the table of values of a function f
template <int N, int M>
DEVICE_DECL_SPEC HOST_DECL_SPEC
numeric_t interpolate(const numeric_t x, const numeric_t y, const numeric_t (&xpoints)[N], const numeric_t (&ypoints)[M],
                      ChemPotTable::matrix_t fvalues) {
    int i = index_interpolation_point(x, xpoints);
    int j = index_interpolation_point(y, ypoints);
    if (i+1 == N) {
        // TODO: this was already an error in the original script
        //  not sure why it's here in the first place. the branch
        //  never evalutes to true
        assert(false);
    }
    assert(i >= 0 and j >= 0); // '{0} <= {1} <= {2} or {3} <= {4} <= {5}'.format(xpoints[i], x, xpoints[i+1], ypoints[i], y, ypoints[i+1]);
    numeric_t xl = xpoints[i]; numeric_t xr = xpoints[i+1];
    numeric_t yb =  ypoints[j]; numeric_t yt = ypoints[j+1];
    numeric_t flb = fvalues(i, j); numeric_t frb = fvalues(i+1, j); numeric_t flt = fvalues(i, j+1);
    numeric_t f = flb + (x - xl)/(xr - xl) * (frb - flb) + (y - yb)/(yt - yb) * (flt - flb);
    return f;
}

// Define the function that calculates the standard chemical potentials of all
// species at given temperature and pressure.
// Parameters:
//   - T is temperature in units of K
//   - P is pressure in units of Pa
// Return:
//   - an array with the standard chemical potentials of all species (in J/mol)
DEVICE_DECL_SPEC HOST_DECL_SPEC
Vector<numeric_t, common::num_species> standard_chemical_potentials(const numeric_t T, const numeric_t P) {
    // Create an array to store the standard chemical potentials of all species
    Vector<numeric_t, common::num_species> u0;
    u0.setConstant(0);

    // Alias to the temperature coordinates of the interpolation tables
    auto& ts = u0_table.temperatures;

    // Alias to the pressure coordinates of the interpolation tables
    auto& ps = u0_table.pressures;

    // Loop over all indices and species names
    for (int i=0; i<num_species; ++i) {
        // Interpolate the standard chemical potential of species i at (T, P)
        u0[i] = userefpotentials ? u0_table[SPECIES_INDEX_VALIDATE(species[i], i)](0, 0)
                                 : interpolate(T, P, ts, ps, u0_table[SPECIES_INDEX_VALIDATE(species[i], i)]);
    }

    // Return the array with standard chemical potentials of the species
    return u0;
}

// Define the function that calculates the chemical potentials of all species
// at given temperature, pressure, and mole amounts of species.
// Parameters:
//   - T is temperature in units of K
//   - P is pressure in units of Pa
//   - n is an array with the mole amounts of the species
// Return:
//   - an array with the chemical potentials of all species (in J/mol)
DEVICE_DECL_SPEC HOST_DECL_SPEC
Vector<numeric_t, common::num_species> chemical_potentials(const ThermodynamicProperties& thermo_props, const Vector<numeric_t, common::num_species>& n) {
    const numeric_t T = thermo_props.T;
    const numeric_t P = thermo_props.P;

    // Calculate the activities of all species at (T, P, n)
    auto ln_a = ln_activities(T, P, n);

    // Return the chemical potentials of all species at (T, P, n)
    return thermo_props.standard_chemical_potentials() + R*T*ln_a;
}

// Define the function that calculates the partial molar derivatives of the
// chemical potentials of the species.
DEVICE_DECL_SPEC HOST_DECL_SPEC
common::hessian_t chemical_potentials_ddn(const ThermodynamicProperties& thermo_props, const Vector<numeric_t, common::num_species>& n) {
    const numeric_t T = thermo_props.T;
    const numeric_t P = thermo_props.P;

    return R*T*ln_activities_ddn(T, P, n);
}

// Define the function that calculates:
//  - G, the Gibbs energy of the system;
//  - u, the gradient of the Gibbs energy, or chemical potentials of the species
//  - H, the Hessian of the Gibbs energy, or partial molar derivatives of activities
// These quantities are normalized by RT for numerical reasons.
DEVICE_DECL_SPEC HOST_DECL_SPEC
ObjectiveResult gibbs_energy(const ThermodynamicProperties& thermo_props, const Vector<numeric_t, common::num_species>& n) {
    const numeric_t T = thermo_props.T;
    const numeric_t RT = R*T;

    ObjectiveResult result;
    result.H = chemical_potentials_ddn(thermo_props, n)/RT;
    result.g = chemical_potentials(thermo_props, n)/RT;
    result.f = n.dot(result.g)/RT;
    return result;
}

// Define the function that calculates the Gibbs energy function for a chemical
// system in which all species are assumed as pure phases. This is needed for
// calculation of an initial guess for the species amounts that satisfies the
// mass conservation conditions.
DEVICE_DECL_SPEC HOST_DECL_SPEC
ObjectiveResult gibbs_energy_pure_phases(const ThermodynamicProperties& thermo_props, const Vector<numeric_t, common::num_species>& n) {
    const numeric_t T = thermo_props.T;
    const numeric_t RT = R*T;

    ObjectiveResult result;
    result.g = thermo_props.standard_chemical_potentials()/RT; // g
    result.H.setConstant(0); // H
    result.f = n.dot(result.g)/RT; // u
    return result;
}

DEVICE_DECL_SPEC HOST_DECL_SPEC
ObjectiveResult gibbs_energy_optimized(const ThermodynamicProperties& thermo_props, const Vector<numeric_t, common::num_species>& n) {
    const numeric_t T = thermo_props.T;
    const numeric_t P = thermo_props.P;
    const numeric_t RT = R*T;

    ObjectiveResult result;

    /*
     * calculate Hessian
     */
    result.H.setConstant(0);

    // The molar amount of H2O(l)
    numeric_t nH2O = n[iH2O];
    numeric_t nH2O_inv = 1/n[iH2O];

    // The mole fraction of H2O(l)
    numeric_t xH2O = nH2O/n.segment(slice_aqueous[0], slice_aqueous[1]-slice_aqueous[0]).sum();

    for (int i=slice_aqueous[0]; i<slice_aqueous[1]; ++i) {
        result.H(i, i) = 1/n[i];
    }
    for (int i=slice_aqueous[0]; i<slice_aqueous[1]; ++i) {
        result.H(i, iH2O) = -nH2O_inv;
    }
    for (int i=slice_aqueous[0]; i<slice_aqueous[1]; ++i) {
        result.H(iH2O, i) = -nH2O_inv * xH2O/(xH2O - 1.0);
    }
    for (int i=slice_aqueous[0]; i<slice_aqueous[1]; ++i) {
        result.H(iH2O, iH2O) = -nH2O_inv;
    }

    /*
     * calculate gradient
     */
    auto& ln_a = result.g; // store the activities directly in the gradient
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
        ln_a[slice_gaseous[0]] += useideal ? 0.0 : thermo_props.ln_fugacity_c02;
        ///////////////////////////////////////////////////////////////////////////

        // Calculate the ln activities of the gaseous species
        ln_a.segment(slice_gaseous[0], slice_gaseous[1]-slice_gaseous[0]) += std::log(P/Pref)*Vector<numeric_t, common::num_species_gaseous>::Ones();
    }

    result.g += thermo_props.standard_chemical_potentials()/RT;
    result.f = n.dot(result.g)/RT;
    return result;
}

#ifdef USE_CUDA
void gibbs_energy_batch_gpu(ThermodynamicProperties thermo_props,
                        int nevals,
                        Vector<numeric_t, common::num_species>* xs,
                        ObjectiveResult* results) {
    CudaExecutionConfiguration launch_conf("gibbs_energy_kernel_simple", (void*) gibbs_energy_kernel_simple, nevals/8);
    chemistry::gibbs_energy_kernel_simple<<<launch_conf.grid_size, launch_conf.block_size>>>(thermo_props, nevals, xs, results);
    cudaDeviceSynchronize();
}
#endif

}
