#ifndef REAKTIVETRANSPORTGPU_CHEM_POT_TABLE_H
#define REAKTIVETRANSPORTGPU_CHEM_POT_TABLE_H

// Not a nice way to code this, but we'll want to keep it simple in the beginning
struct ChemPotTable {
    using matrix_t = Eigen::Matrix<numeric_t, 4, 3>;

    // The temperatures where the standard chemical potentials are available (in K)
    numeric_t temperatures[4];
    numeric_t pressures[3] = {1e5, 50e5, 100e5};

    matrix_t u0_table[common::num_species];

    DEVICE_DECL_SPEC HOST_DECL_SPEC
    ChemPotTable() {
        // Initialize temperature
        uint8_t tmp[4] = { 25, 50, 75, 100 };
        for (size_t i=0; i<4; ++i) {
            temperatures[i] = tmp[i]+273.15;
        }

        // Initialize the actual table
        // The standard chemical potentials of species H2O(l) (in J/mol)
        u0_table[SPECIES_INDEX_VALIDATE("H2O(l)", 0)] <<
                -237181.72, -237093.28, -237003.23,
                -239006.71, -238917.46, -238826.59,
                -240977.57, -240887.12, -240795.03,
                -999999999, -242992.18, -242898.53;

        // The standard chemical potentials of species H+(aq) (in J/mol)
        u0_table[SPECIES_INDEX_VALIDATE("H+(aq)", 1)] <<
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0;

        // The standard chemical potentials of species OH-(aq) (in J/mol)
        u0_table[SPECIES_INDEX_VALIDATE("OH-(aq)", 2)] <<
                -157297.48, -157320.08, -157342.16,
                -156905.20, -156924.41, -156943.13,
                -156307.81, -156328.46, -156348.60,
                -999999999, -155558.52, -155583.77;

        // The standard chemical potentials of species HCO3-(aq) (in J/mol)
        u0_table[SPECIES_INDEX_VALIDATE("HCO3-(aq)", 3)] <<
                -586939.89, -586820.91, -586698.78,
                -589371.68, -589247.85, -589120.88,
                -591758.94, -591634.79, -591507.48,
                -999999999, -593983.38, -593858.95;

        // The standard chemical potentials of species CO3--(aq) (in J/mol)
        u0_table[SPECIES_INDEX_VALIDATE("CO3--(aq)", 4)] <<
                -527983.14, -528011.90, -528039.31,
                -526461.60, -526494.20, -526525.56,
                -524473.15, -524514.70, -524555.01,
                -999999999, -522121.51, -522175.91;

        // The standard chemical potentials of species CO2(aq) (in J/mol)
        u0_table[SPECIES_INDEX_VALIDATE("CO2(aq)", 5)] <<
                -385974.00, -385813.47, -385650.13,
                -389147.04, -388979.61, -388809.39,
                -392728.17, -392556.66, -392382.38,
                -999999999, -396484.88, -396307.90;

        // The standard chemical potentials of species Na+(aq) (in J/mol)
        u0_table[SPECIES_INDEX_VALIDATE("Na+(aq)", 6)] <<
                -261880.74, -261886.18, -261890.73,
                -263384.24, -263384.84, -263384.60,
                -264979.92, -264978.35, -264975.97,
                -999999999, -266664.71, -266661.71;

        // The standard chemical potentials of species Cl-(aq) (in J/mol)
        u0_table[SPECIES_INDEX_VALIDATE("Cl-(aq)", 7)] <<
                -131289.74, -131204.66, -131117.63,
                -132590.59, -132504.28, -132416.09,
                -133682.46, -133598.32, -133512.30,
                -999999999, -134501.87, -134420.82;

        // The standard chemical potentials of species Ca++(aq) (in J/mol)
        u0_table[SPECIES_INDEX_VALIDATE("Ca++(aq)", 8)] <<
                -552790.08, -552879.51, -552968.88,
                -551348.98, -551437.62, -551526.28,
                -549855.33, -549945.95, -550036.62,
                -999999999, -548400.71, -548495.68;

        // The standard chemical potentials of species CO2(g) (in J/mol)
        u0_table[SPECIES_INDEX_VALIDATE("CO2(g)", 9)] <<
                -394358.74, -394358.74, -394358.74,
                -399740.71, -399740.71, -399740.71,
                -405197.73, -405197.73, -405197.73,
                -410726.87, -410726.87, -410726.87;

        // The standard chemical potentials of species CaCO3(s,calcite) (in J/mol)
        u0_table[SPECIES_INDEX_VALIDATE("CaCO3(s,calcite)", 10)] <<
                -1129177.92, -1128996.94, -1128812.26,
                -1131580.05, -1131399.06, -1131214.39,
                -1134149.89, -1133968.91, -1133784.23,
                -1136882.60, -1136701.62, -1136516.94;
    }

    DEVICE_DECL_SPEC HOST_DECL_SPEC
    const matrix_t& operator[](size_t ispecies) const {
        return u0_table[ispecies];
    }
};

#endif //REAKTIVETRANSPORTGPU_CHEM_POT_TABLE_H
