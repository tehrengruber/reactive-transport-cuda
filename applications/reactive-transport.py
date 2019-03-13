# Import the Python package Numpy so we can perform linear algebra calculations
from numpy import *

# Import copy package to make deep copies of EquilibriumState objects
import copy

# Import matplotlib for the plots
import matplotlib.pyplot as plt

# Configure the plot styles
plt.style.use('ggplot')
plt.rc('font', size=16)
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rc('lines', linewidth=4)
plt.rc('figure', autolayout=True)

# The universal gas constant in units of J/(mol*K)
R = 8.3144598

# The reference pressure P(ref) (in units of Pa)
Pref = 1.0e5

# The Drummond model for ln activity coefficient of CO2(aq).
# Parameters:
# - T is temperature in K
# - I is ionic strength in molal
# Return: the ln activity coefficient of CO2(aq)
def ln_activity_coeff_co2_drummond(T, I):
    c1 = -1.0312
    c2 = 1.2806e-3
    c3 = 255.9
    c4 = 0.4445
    c5 = -1.606e-3
    return (c1 + c2*T + c3/T)*I - (c4 + c5*T)*I/(1 + I)

# The Davies model for the ln activity coefficient of aqueous ions.
# Parameters:
# - Z is the charge of the ionic species
# - I is the ionic strength in molal
# Return: the ln of activity coefficient of the aqueous ion
def ln_activity_coeff_ion_davies(Z, I):
    Agamma = 0.5095
    sqrtI = sqrt(I)
    return 2.30258 * (-Agamma * Z**2 * (sqrtI / (1.0 + sqrtI) - 0.3 * I))

# Define a function that solves the nonlinear equation f(x) = 0.
# Parameters:
# - f is the function
# - fprime is the function first0order derivative
# - x0 is the initial guess
# Return: the value of x such that f(x) = 0
def newton(f, fprime, x0):
    maxiters = 100 # maximum number of iterations
    tolerance = 1e-4 # the tolerance for the convergence
    counter = 0 # the counter of number of iterations
    x = x0 # start with the solution x being the initial guess
    # Perform one or more Newton iterations
    for counter in xrange(maxiters):
        x = x - f(x) / fprime(x) # calculate the new approximation for x
        if abs(f(x)) < tolerance: # check for convergence
            return x # return x if the calculation converged
    # Raise an error if the calculation did not converge.
    raise RuntimeError('Could not calculate the \
        solution of the nonlinear equation in %d iterations.' % counter)

# The Peng-Robinson model for the ln fugacity coefficient of CO2(g).temperature
# Parameters:
# - T is temperature in K
# - P is pressure in Pa.
# Return: the ln of fugacity coefficient of CO2(g)
def ln_fugacity_coefficient_co2(T, P):
    # The critical temperature and pressure of CO2
    Tc = 304.2    # CO2 critical temperature in K
    Pc = 73.83e5  # CO2 critical pressure in Pa
    # The acentric factor of CO2
    omega = 0.224
    # Create variables for the parameters of Peng-Robinson EOS
    epsilon = 1 - sqrt(2.0)
    sigma = 1 + sqrt(2.0)
    Omega = 0.07780
    Psi = 0.45724
    # The reduced temperature, Tr, and reduced pressure, Pr
    Tr = T / Tc
    Pr = P / Pc
    # The evaluation of the alpha(Tr, omega) function
    alpha = (1 + (0.37464 + 1.54226*omega - 0.26992*omega**2)*(1 - sqrt(Tr)))**2
    # The beta and q contants
    beta = Omega * Pr / Tr
    q = Psi/Omega * alpha/Tr
    # Define the function f that represents the nonlinear equation f(x) = 0
    def f(Z):
        return (1 + beta - q*beta*(Z - beta)/((Z + epsilon*beta)*(Z + sigma*beta))) - Z
    # Define the first order derivative of function f'(x)
    def fprime(Z):
        aux = (Z + epsilon*beta)*(Z + sigma*beta)
        return -q*beta/aux * (1.0 - (Z - beta)*(2*Z + (epsilon + sigma)*beta)/aux) - 1
    # Calculate the compressibility factor of CO2 at (T, P)
    Z0 = 1.0 # the initial guess for the compressibility factor
    Z = newton(f, fprime, Z0) # use newton function to perform the calculation of Z
    # Calcute theta
    theta = 1.0/(sigma - epsilon) * log((Z + sigma*beta)/(Z + epsilon*beta))
    # Calculate ln_phiCO2
    ln_phiCO2 = Z - 1 - log(Z - beta) - q*theta

    return ln_phiCO2

# Create a list of component names
# components = ['H', 'O', 'C', 'Na', 'Cl', 'Ca', 'Z']
components = ['H', 'O', 'C', 'Na', 'Cl', 'Ca']

# Create a list of species names
species = ['H2O(l)',
           'H+(aq)',
           'OH-(aq)',
           'HCO3-(aq)',
           'CO3--(aq)',
           'CO2(aq)',
           'Na+(aq)',
           'Cl-(aq)',
           'Ca++(aq)',
           'CO2(g)',
           'CaCO3(s,calcite)']

# Create a list with the charges of the aqueous species
charges = array([0, 1, -1, -1, -2, 0, 1, -1, 2])

# The number of components and species
num_components = len(components)
num_species = len(species)

# The number of aqueous, gaseous, and mineral species
num_species_aqueous = 9
num_species_gaseous = 1
num_species_mineral = 1

slice_aqueous = slice(0, num_species_aqueous)
slice_gaseous = slice(num_species_aqueous, num_species_aqueous + num_species_gaseous)
slice_mineral = slice(num_species_aqueous + num_species_gaseous, -1)

# Store the index of H2O(l)
iH2O = species.index('H2O(l)')

# Store the index of H+(aq)
iH = species.index('H+(aq)')

# Store the index of CO2(aq)
iCO2 = species.index('CO2(aq)')

# Create a list with the indices of the charged species
icharged = [idx for idx, charge in enumerate(charges) if charge != 0.0]

# Construct the formula matrix of the chemical system
formula_matrix = array([
    [2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # H
    [1, 0, 1, 3, 3, 2, 0, 0, 0, 2, 3],  # O
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],  # C
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Na
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Cl
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # Ca
    # [0, 1,-1,-1,-2, 0, 1,-1, 2, 0, 0]   # Z
])

# Create an array with the molar masses of the components
components_molar_masses = array([
    1.0079,   # H
    15.9994,  # O
    12.0107,  # C
    22.9898,  # Na
    35.453,   # Cl
    40.078])  # Ca

# Create an array with the molar masses of the species
species_molar_masses = transpose(formula_matrix).dot(components_molar_masses)

# A global variable to allow one to choose if ideal activity models are used
useideal = False

# A global variable to allow one to choose only reference state standard
# chemical potentials are used, without interpolation
userefpotentials = False

# Define the function that calculates the activities of all aqueous species.
# Parameters:
#   - T is temperature in units of K
#   - P is pressure in units of Pa
#   - nphase is an array with the mole amounts of aqueous species only
# Return:
#   - an array with the ln activities of all aqueous species
def ln_activities_aqueous_species(T, P, nphase):

    # Calculate the molalities of the aqueous species and their natural log
    m = 55.508 * nphase / nphase[iH2O]

    # Calculate the ionic strength of the aqueous phase
    I = 0.5 * sum([mi*Zi**2 for mi, Zi in zip(m, charges)])

    # Calculate the mole fraction of water species, H2O(l)
    xH2O = nphase[iH2O] / sum(nphase)

    # Create an array to store the ln activity coeffs of the aqueous species
    ln_g = zeros(num_species_aqueous)

    ###########################################################################
    # Calculate the ln activity coefficient of CO2(aq)
    # ---- USE DRUMMOND (1981) MODEL ----
    ln_g[iCO2] = 0.0 if useideal else ln_activity_coeff_co2_drummond(T, I)
    ###########################################################################

    ###########################################################################
    # Calculate the ln activity coefficient of each charged species
    # ---- USE DAVIES MODEL FOR EACH CHARGED SPECIES ----
    for i in icharged:
        ln_g[i] = 0.0 if useideal else ln_activity_coeff_ion_davies(charges[i], I)
    ###########################################################################

    # Calculate the natural log of the molalities of the aqueous species
    ln_m = log(m)

    # Calculate the ln activities of the solute species
    ln_a = ln_g + ln_m

    # Calculate the ln activity of the solvent species H2O(l)
    ln_a[iH2O] = -(1 - xH2O)/xH2O

    # Return the array ln_a with the ln activities of the aqueous species
    return ln_a


# Define the function that calculates the activities of all gaseous species.
# Parameters:
#   - T is temperature in units of K
#   - P is pressure in units of Pa
#   - nphase is an array with the mole amounts of gaseous species only
# Return:
#   - an array with the ln activities of all gaseous species
def ln_activities_gaseous_species(T, P, nphase):
    # Calculate the array of mole fractions of the gaseous species
    x = nphase/sum(nphase)

    # Create an array to store the ln fugacity coeffs of the gaseous species
    ln_phi = zeros(num_species_gaseous)

    ###########################################################################
    # Calculate the ln fugacity coefficient of the only gaseous species, CO2(g)
    # ---- USE PENG-ROBINSON (1976) MODEL ----
    ln_phi[0] = 0.0 if useideal else ln_fugacity_coefficient_co2(T, P)
    ###########################################################################

    # Calculate the array of ln mole fractions of the gaseous species
    ln_x = log(x)

    # Calculate the ln activities of the gaseous species
    ln_a = ln_phi + ln_x + log(P/Pref)

    # Return the array ln_a with the ln activities of the gaseous species
    return ln_a


# Define the function that calculates the activities of the mineral species.
# Parameters:
#   - T is temperature in units of K
#   - P is pressure in units of Pa
#   - nphase is an array with the mole amounts of the mineral species
# Return:
#   - an array with the ln activities of mineral species
def ln_activities_mineral_species(T, P, nphase):
    return zeros(num_species_mineral)


def ln_activities_aqueous_species_ddn(T, P, nphase):
    # The molar amount of H2O(l)
    nH2O = nphase[iH2O]

    # The mole fraction of H2O(l)
    xH2O = nH2O/sum(nphase)

    # Calculate the partial molar derivatives of the solute activities
    ddn = diag(1.0/nphase)
    ddn[:, iH2O] = -1.0/nH2O

    # Calculate the partial molar derivatives of the solvent activity
    ddn[iH2O, :] = -1.0/nH2O * xH2O/(xH2O - 1.0)
    ddn[iH2O, iH2O] = -1.0/nH2O

    return ddn


# Define the function that calculates the activities of all species.
# Parameters:
#   - T is temperature in units of K
#   - P is pressure in units of Pa
#   - n is an array with the mole amounts of all species in the chemical system
# Return:
#   - an array with the ln activities of all species in the chemical system
def ln_activities(T, P, n):
    # Create slices of n corresponding to each phase
    n_aqueous = n[slice_aqueous]
    n_gaseous = n[slice_gaseous]
    n_mineral = n[slice_mineral]

    # Calculate the activities of aqueous, gaseous and mineral species and
    # concatenate them into a single array with the ln activities of all
    # species in the chemical system.
    ln_a = zeros(num_species)
    ln_a[slice_aqueous] = ln_activities_aqueous_species(T, P, n_aqueous)
    ln_a[slice_gaseous] = ln_activities_gaseous_species(T, P, n_gaseous)
    ln_a[slice_mineral] = ln_activities_mineral_species(T, P, n_mineral)

    return ln_a

# Define the function that calculates the partial molar derivatives of the
# ln activities of all species
def ln_activities_ddn(T, P, n):
    # Create an array with the entries in n corresponding to aqueous species
    n_aqueous = n[slice_aqueous]

    # The matrix with partial molar derivatives of the activities
    ln_a_ddn = zeros((num_species, num_species))

    ln_a_ddn[slice_aqueous, slice_aqueous] = \
        ln_activities_aqueous_species_ddn(T, P, n_aqueous)

    return ln_a_ddn

# Define the function that calculates the concentrations of the species.
# For aqueous solutes, concentrations are molalities.
# For aqueous solvent H2O(l), mole fraction.
# For gaseous species, partial pressure.
# For mineral species, mole fractions.
def concentrations(T, P, n):
    c = ones(num_species)
    c[slice_aqueous] = 55.508 * n[slice_aqueous]/n[iH2O]
    c[iH2O] = n[iH2O]/sum(n[slice_aqueous])
    c[slice_gaseous] = n[slice_gaseous]/sum(n[slice_gaseous]) * P/Pref
    c[slice_mineral] = 1
    return c


# Define the function that calculates the masses of all species.
def masses(n):
    return species_molar_masses * n


# Define the function that calculates the amounts of elements
def elementAmounts(n):
    return formula_matrix.dot(n)

# Return the amounts of elements in the aqueous phase
def elementAmountsAqueous(n):
    return formula_matrix[:, slice_aqueous].dot(n[slice_aqueous])

# Return the amounts of elements in the gaseous phase
def elementAmountsGaseous(n):
    return formula_matrix[:, slice_gaseous].dot(n[slice_gaseous])

# Return the amounts of elements in the mineral phase
def elementAmountsMineral(n):
    return formula_matrix[:, slice_mineral].dot(n[slice_mineral])

# Create a dictionary to store the standard chemical potentials of all species
u0_table = {}

# The temperatures where the standard chemical potentials are available (in K)
u0_table['temperatures'] = array([25, 50, 75, 100]) + 273.15

# The pressures where the standard chemical potentials are available (in Pa)
u0_table['pressures'] = array([1, 50, 100]) * 1e5

# The standard chemical potentials of species H2O(l) (in J/mol)
u0_table['H2O(l)'] = array([
    [-237181.72, -237093.28, -237003.23],
    [-239006.71, -238917.46, -238826.59],
    [-240977.57, -240887.12, -240795.03],
    [-999999999, -242992.18, -242898.53],
])

# The standard chemical potentials of species H+(aq) (in J/mol)
u0_table['H+(aq)'] = array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])

# The standard chemical potentials of species OH-(aq) (in J/mol)
u0_table['OH-(aq)'] = array([
    [-157297.48, -157320.08, -157342.16],
    [-156905.20, -156924.41, -156943.13],
    [-156307.81, -156328.46, -156348.60],
    [-999999999, -155558.52, -155583.77],
])

# The standard chemical potentials of species HCO3-(aq) (in J/mol)
u0_table['HCO3-(aq)'] = array([
    [-586939.89, -586820.91, -586698.78],
    [-589371.68, -589247.85, -589120.88],
    [-591758.94, -591634.79, -591507.48],
    [-999999999, -593983.38, -593858.95],
])

# The standard chemical potentials of species CO3--(aq) (in J/mol)
u0_table['CO3--(aq)'] = array([
    [-527983.14, -528011.90, -528039.31],
    [-526461.60, -526494.20, -526525.56],
    [-524473.15, -524514.70, -524555.01],
    [-999999999, -522121.51, -522175.91],
])

# The standard chemical potentials of species CO2(aq) (in J/mol)
u0_table['CO2(aq)'] = array([
    [-385974.00, -385813.47, -385650.13],
    [-389147.04, -388979.61, -388809.39],
    [-392728.17, -392556.66, -392382.38],
    [-999999999, -396484.88, -396307.90],
])

# The standard chemical potentials of species Na+(aq) (in J/mol)
u0_table['Na+(aq)'] = array([
    [-261880.74, -261886.18, -261890.73],
    [-263384.24, -263384.84, -263384.60],
    [-264979.92, -264978.35, -264975.97],
    [-999999999, -266664.71, -266661.71],
])

# The standard chemical potentials of species Cl-(aq) (in J/mol)
u0_table['Cl-(aq)'] = array([
    [-131289.74, -131204.66, -131117.63],
    [-132590.59, -132504.28, -132416.09],
    [-133682.46, -133598.32, -133512.30],
    [-999999999, -134501.87, -134420.82],
])

# The standard chemical potentials of species Ca++(aq) (in J/mol)
u0_table['Ca++(aq)'] = array([
    [-552790.08, -552879.51, -552968.88],
    [-551348.98, -551437.62, -551526.28],
    [-549855.33, -549945.95, -550036.62],
    [-999999999, -548400.71, -548495.68],
])

# The standard chemical potentials of species CO2(g) (in J/mol)
u0_table['CO2(g)'] = array([
    [-394358.74, -394358.74, -394358.74],
    [-399740.71, -399740.71, -399740.71],
    [-405197.73, -405197.73, -405197.73],
    [-410726.87, -410726.87, -410726.87],
])

# The standard chemical potentials of species CaCO3(s,calcite) (in J/mol)
u0_table['CaCO3(s,calcite)'] = array([
    [-1129177.92, -1128996.94, -1128812.26],
    [-1131580.05, -1131399.06, -1131214.39],
    [-1134149.89, -1133968.91, -1133784.23],
    [-1136882.60, -1136701.62, -1136516.94],
])

# Find the fist index i such that values[i] <= val <= values[i+1].
# Parameters:
# - val is the given value
# - values is the list of sorted values in whic
# Example:
# >>> print index_interpolation_point(60.0, [0.0, 25.0, 50.0, 100.0, 150.0])
# 3
# >>> print index_interpolation_point(150.0, [0.0, 25.0, 50.0, 100.0, 150.0])
# 3
def index_interpolation_point(val, values):
    for i, v in enumerate(values):
        if v > val:
            return i-1
        if v == val: return i if i+1 < len(values) else i-1
    return -1

# Define the interpolation function.
# Parameters:
#   - x is the x-coordinate where the interpolation is performed
#   - y is the y-coordinate where the interpolation is performed
#   - xpoints is an array with the x-coordinates where fvalues are available
#   - ypoints is an array with the y-coordinates where fvalues are available
#   - fvalues is a 2D array with the table of values of a function f
def interpolate(x, y, xpoints, ypoints, fvalues):
    i = index_interpolation_point(x, xpoints)
    j = index_interpolation_point(y, ypoints)
    if i+1 == len(xpoints): return fva
    assert i >= 0 and j >= 0, '{0} <= {1} <= {2} or {3} <= {4} <= {5}'.format(
        xpoints[i], x, xpoints[i+1], ypoints[i], y, ypoints[i+1])
    xl, xr = xpoints[i], xpoints[i+1]
    yb, yt = ypoints[j], ypoints[j+1]
    flb, frb, flt = fvalues[i][j], fvalues[i+1][j], fvalues[i][j+1]
    f = flb + (x - xl)/(xr - xl) * (frb - flb) + (y - yb)/(yt - yb) * (flt - flb)
    return f

# Define the function that calculates the standard chemical potentials of all
# species at given temperature and pressure.
# Parameters:
#   - T is temperature in units of K
#   - P is pressure in units of Pa
# Return:
#   - an array with the standard chemical potentials of all species (in J/mol)
def standard_chemical_potentials(T, P):
    # Create an array to store the standard chemical potentials of all species
    u0 = zeros(num_species)

    # Alias to the temperature coordinates of the interpolation tables
    ts = u0_table['temperatures']

    # Alias to the pressure coordinates of the interpolation tables
    ps = u0_table['pressures']

    # Loop over all indices and species names
    for i, name in enumerate(species):
        # Interpolate the standard chemical potential of species i at (T, P)
        u0[i] = u0_table[name][0][0] if userefpotentials else \
            interpolate(T, P, ts, ps, u0_table[name])

    # Return the array with standard chemical potentials of the species
    return u0


# Define the function that calculates the chemical potentials of all species
# at given temperature, pressure, and mole amounts of species.
# Parameters:
#   - T is temperature in units of K
#   - P is pressure in units of Pa
#   - n is an array with the mole amounts of the species
# Return:
#   - an array with the chemical potentials of all species (in J/mol)
def chemical_potentials(T, P, n):
    # Calculate the standard chemical potentials of all species at (T, P)
    u0 = standard_chemical_potentials(T, P)

    # Calculate the activities of all species at (T, P, n)
    ln_a = ln_activities(T, P, n)

    # Return the chemical potentials of all species at (T, P, n)
    return u0 + R*T*ln_a


# Define the function that calculates the partial molar derivatives of the
# chemical potentials of the species.
def chemical_potentials_ddn(T, P, n):
    return R*T*ln_activities_ddn(T, P, n)


# Define the function that calculates:
#  - G, the Gibbs energy of the system;
#  - u, the gradient of the Gibbs energy, or chemical potentials of the species
#  - H, the Hessian of the Gibbs energy, or partial molar derivatives of activities
# These quantities are normalized by RT for numerical reasons.
def gibbs_energy(T, P, n):
    RT = R*T
    u = chemical_potentials(T, P, n)/RT
    H = chemical_potentials_ddn(T, P, n)/RT
    G = n.dot(u)/RT
    return G, u, H

# Define the function that calculates the Gibbs energy function for a chemical
# system in which all species are assumed as pure phases. This is needed for
# calculation of an initial guess for the species amounts that satisfies the
# mass conservation conditions.
def gibbs_energy_pure_phases(T, P, n):
    RT = R*T
    u = standard_chemical_potentials(T, P)/RT
    H = zeros((num_species, num_species))
    G = n.dot(u)/RT
    return G, u, H


class ObjectiveResult:
    def __init__(self):
        self.f = None  # the value of the objective function at x
        self.g = None  # the gradient of the objective function at x
        self.H = None  # the Hessian of the objective function at x


class OptimumProblem:
    def __init__(self):
        self.objective = None  # the objective function
        self.A = None  # the matrix A in the equality constraints
        self.b = None  # the vector b in the equality constraints


class OptimumState:
    def __init__(self):
        self.x = None  # the variables x = (x1, ..., xn)
        self.y = None  # the Lagrange multipliers y = (y1, ..., ym)
        self.z = None  # the Lagrange multipliers z = (z1, ..., zn)


def minimize(state, problem, **options):
    """Minimize a function f(x) subject to constraints Ax = b and x >= 0."""

    imax = options.get('imax', 100)
    mu = options.get('mu', 1.0e-14)
    tau = options.get('tau', 0.99999)
    tol = options.get('tol', 1.0e-6)
    output = options.get('output', True)

    if output: output = open('minimize-output.txt', 'a+')

    A = problem.A
    b = problem.b

    m = A.shape[0]
    n = A.shape[1]
    p = m + n
    t = m + 2 * n

    if state.x is None:
        state.x = full(n, mu)

    if state.y is None:
        state.y = zeros(m)

    if state.z is None:
        state.z = ones(n)

    x = state.x
    y = state.y
    z = state.z

    F = zeros(t)
    J = zeros((t, t))

    f = 0.0
    g = zeros(n)  # F[:n]
    H = zeros((n, n))  # J[:n, :n]()

    if output:
        header = '{:20} '.format('Iteration')
        header += '{:20} '.format('Error')
        for i in xrange(n):
            header += '{:20} '.format('x[%s]' % species[i])
        for i in xrange(m):
            header += '{:20} '.format('y[%s]' % components[i])
        for i in xrange(n):
            header += '{:20} '.format('z[%s]' % species[i])
        bar = '=' * len(header)
        header = bar + '\n' + header + '\n' + bar
        print >>output, header

    for it in xrange(imax):
        f, g, H = problem.objective(x)

        # Assemble the negative of the residual vector -F
        #     [g(x) - tr(A)*y - z]
        # F = [      A*x - b     ]
        #     [    X*Z*e - mu    ]
        F[:n] = g - A.T.dot(y) - z
        F[n:p] = A.dot(x) - b
        F[-n:] = x * z - mu

        # Calculate the optimality, feasibility, complementarity errors
        error_opt = linalg.norm(F[:n], inf)
        error_fea = linalg.norm(F[n:p], inf)
        error_com = linalg.norm(F[-n:], inf)

        # Calculate the current total error
        error = max([error_opt, error_fea, error_com])

        if output:
            string = '{:20} '.format(str(it))
            string += '{:20} '.format(str(error))
            for i in xrange(n):
                string += '{:20} '.format(str(x[i]))
            for i in xrange(m):
                string += '{:20} '.format(str(y[i]))
            for i in xrange(n):
                string += '{:20} '.format(str(z[i]))
            print >>output, string

        # Check if the calculation has converged
        if error < tol:
            break

        # Assemble the Jacoabian matrix J
        #     [H -tr(A) -I]
        # J = [A    0    0]
        #     [Z    0    X]
        J[:n, :n] = H
        J[:n, n:p] = -A.T
        J[:n, -n:] = -eye(n)
        J[n:p, :n] = A
        J[-n:, :n] = diag(z)
        J[-n:, -n:] = diag(x)

        mstr = ""
        for i in xrange(0, t):
            for j in xrange(0, t):
                mstr = mstr + str(J[i, j]).rjust(4) + " "
            mstr = mstr + "\n"
        #print(mstr)

        # Compute the Newton step d = [dx dy dx]
        delta = linalg.solve(J, -F)

        dx = delta[:n]
        dy = delta[n:p]
        dz = delta[-n:]

        # Calculate the new values for x and z
        for i in xrange(n):
            x[i] += dx[i] if x[i] + dx[i] > 0.0 else -tau * x[i]
            z[i] += dz[i] if z[i] + dz[i] > 0.0 else -tau * z[i]

        # Calculate the new values for y
        y += dy

    return (it, it < imax, error)


# Auxiliary funtion that calculates the vector b of component amounts for
# a given recipe involving H2O, CO2, NaCl, and CaCO3.
def component_amounts(kgH2O, molCO2, molNaCl, molCaCO3):
    molH2O = 55.508 * kgH2O
    b = [2*molH2O,                        # H
         molH2O + 2*molCO2 + 3*molCaCO3,  # O
         molCO2 + molCaCO3,               # C
         molNaCl,                         # Na
         molNaCl,                         # Cl
         molCaCO3]                        # Ca
    # Ensure the amounts of each component is greater than zero
    for i in range(len(b)):
        if b[i] <= 0.0:
            b[i] = 1e-10  # replace zero or negative by a small positive number
    return b


class EquilibriumState:
    def __init__(self):
        self.T = None  # temperature in K
        self.P = None  # pressure in Pa
        self.n = None  # amounts of species in mol
        self.m = None  # masses of species in g
        self.c = None  # concentrations of the species
        self.a = None  # activities of the species
        self.g = None  # activity coefficients of the species
        self.x = None  # alias to amounts of species in mol
        self.y = None  # Lagrange multipliers y in J/mol
        self.z = None  # Lagrange multipliers z in J/mol

    def clone(self):
        return copy.deepcopy(self)


    def __str__(self):
        bar = '='*150+'\n'
        thinbar = '-'*150+'\n'
        res = bar
        res += '{:25} {:25} {:25} {:25}\n'.format(
            'Temperature [K]', 'Temperature [C]',
            'Pressure [Pa]', 'Pressure [bar]')
        res += thinbar
        res += '{:25} {:25} {:25} {:25}\n'.format(
            str(self.T), str(self.T - 273.15),
            str(self.P), str(self.P * 1e-5))
        res += bar
        res += '{:25} {:25} {:25} {:25} {:25} {:25}\n'.format(
            'Species', 'Amount [mol]', 'Mass [g]', 'Concentration',
            'Activity', 'Activity Coefficient')
        res += thinbar
        for i, name in enumerate(species):
            res += '{:25} {:25} {:25} {:25} {:25} {:25}\n'.format(
                name, str(self.n[i]), str(self.m[i]), str(self.c[i]),
                str(self.a[i]), str(self.g[i]))
        res += bar
        return res


# Calculate the equilibrium state of the chemical system.
# Parameters:
#   - T is temperature in K
#   - P is pressure in Pa
#   - b is an array with molar amounts of components
#   - options is a dictionary with options for the calculation
# Return:
#   - an state object with members n, y, z, so that the
#     molar amounts of the species is given by state.n,
#     and the Lagrange multipliers by state.y and state.z.
def equilibrate(T, P, b, state=None, **options):
    # Define the minimization problem
    problem = OptimumProblem()
    problem.A = array(formula_matrix)
    problem.b = array(b)

    # Check if state is None, and if so, find an initial guess
    if state is None:
        # Create an EquilibriumState object
        state = EquilibriumState()

        # Define the objective function to calculate an initial guess
        problem.objective = lambda x: gibbs_energy_pure_phases(T, P, x)

        # Minimize the Gibbs energy assuming each species is a pure phase
        minimize(state, problem, **options)
        state.y *= R * T
        state.z *= R * T

    state.y /= R*T
    state.z /= R*T

    # Define now the objective function with a normalized Gibbs energy function
    problem.objective = lambda x: gibbs_energy(T, P, x)

    # Minimize the Gibbs energy of the chemical system
    minimize(state, problem, **options)

    # Finalize the setting of the equilibrium state
    state.T = T
    state.P = P
    state.n = state.x
    state.m = masses(state.x)
    state.y = state.y * R * T
    state.z = state.z * R * T
    state.a = exp(ln_activities(T, P, state.x))
    state.c = concentrations(T, P, state.x)
    state.g = state.a/state.c

    return state

# Auxiliary time related constants
second = 1.0
minute = 60.0
hour = 60 * minute
day = 24 * hour
year = 365 * day

# Solve a tridiagonal matrix equation using Thomas algorithm.
def thomas(a, b, c, d):
    n = len(d)
    c[0] /= b[0]
    for i in xrange(1, n - 1):
        c[i] /= b[i] - a[i]*c[i - 1]
    d[0] /= b[0]
    for i in xrange(1, n):
        d[i] = (d[i] - a[i]*d[i - 1])/(b[i] - a[i]*c[i - 1])
    x = d
    for i in reversed(xrange(0, n - 1)):
        x[i] -= c[i]*x[i + 1]
    return x

# Perform a transport step
def transport(u, dt, dx, v, D, ul):
    n = len(u)
    alpha = D*dt/dx**2
    beta = v*dt/dx
    a = full(n, -beta - alpha)
    b = full(n, 1 + beta + 2*alpha)
    c = full(n, -alpha)
    # b[0] = 1.0
    # c[0] = 0.0
    # u[0] = ul
    u[0] += beta*ul
    b[0] = 1 + beta + alpha
    b[-1] = 1 + beta + alpha
    thomas(a, b, c, u)


def output(state):
    pass


ncells = 100


D  = 0.0      # in units of m2/s
# D  = 1.0e-9      # in units of m2/s
# v  = 1.0/day     # in units of m/s
v  = 1.0/day     # in units of m/s

dx = 1.0/ncells     # in units of m
# dt = 1*hour   # in units of s
dt = 60*minute   # in units of s
# dt = dx**2/D * 0.1  # in units of s

T = 60.0 + 273.15  # 60 degC in K
P = 100 * 1e5      # 100 bar in Pa

b_bc = component_amounts(kgH2O=1.0, molCO2=0.05, molNaCl=0.0, molCaCO3=0.0)
b_ic = component_amounts(kgH2O=1.0, molCO2=0.0, molNaCl=0.7, molCaCO3=0.1)

state_bc = equilibrate(T, P, b_bc)
state_ic = equilibrate(T, P, b_ic)

b0 = zeros((ncells, num_components))
b = zeros((ncells, num_components))
bfluid = zeros((ncells, num_components))
bsolid = zeros((ncells, num_components))

b[:] = elementAmounts(state_ic.n)

states = [state_ic.clone() for _ in xrange(ncells)]

# states[0] = state_bc.clone()

# ylim = None
ylim = (0.0, 1.1)

x = linspace(0.0, 1.0, ncells)

nsteps = 10
t = 0.0
while True:
    # pH = [-log10(state.a[iH]) for state in states]
    nCO2 = [state.n[iCO2] for state in states]
    nCa = [state.n[species.index('Ca++(aq)')] for state in states]
    nCaCO3 = [state.n[-1] for state in states]
    plt.title('Time = %.2f day' % (t/day))
    if ylim is not None: plt.ylim(ylim)
    # plt.plot(x, pH)
    plt.plot(x, nCO2, label='CO2(aq)')
    plt.plot(x, nCa, label='Ca++')
    plt.plot(x, nCaCO3, label='Calcite')
    plt.ylim((0.0, 0.11))
    plt.legend(loc='center right')
    if ylim is None: ylim = plt.ylim()
    plt.pause(0.5)

    # savetxt(output, u, delimiter=',', newline=' ')
    # output.write('\n')

    for icell in xrange(ncells):
        bfluid[icell] = elementAmountsAqueous(states[icell].n)
        # bsolid[icell] = elementAmountsMineral(states[icell].n)
        bsolid[icell] = formula_matrix[:, -1] * states[icell].n[-1]

    # import sys
    # sys.exit()
    
    for j in xrange(num_components):
        transport(bfluid[:, j], dt, dx, v, D, b_bc[j])

    b[:] = bsolid + bfluid

    for icell in xrange(ncells):
        equilibrate(T, P, b[icell], states[icell])

    print("Progress:", linalg.norm(b - b0))

    # if t >= dt*nsteps:
    # if linalg.norm(b - b0) < 1e-2:
    #     break

    b0[:] = b
    t += dt

    plt.clf()


print "Finished!"

plt.show()