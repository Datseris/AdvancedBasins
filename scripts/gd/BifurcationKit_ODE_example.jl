using DrWatson
@quickactivate
using Attractors

include(srcdir("vis", "basins_plotting.jl"))

# Neural mass model as in
# https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/tutorials/ode/tutorialsODE/#Neural-mass-equation-(Hopf-aBS)

function neural_mass_rule(z, p, t)
	@unpack J, α, E0, τ, τD, τF, U0 = p
	E, x, u = z
	SS0 = J * u * x * E + E0
	SS1 = α * log(1 + exp(SS0 / α))
	dz1 = (-E + SS1) / τ
	dz2 =	(1.0 - x) / τD - u * x * E
	dz3 = (U0 - u) / τF +  U0 * (1.0 - u) * E
	SVector(dz1, dz2, dz3)
end

p0 = (α = 1.5, τ = 0.013, J = 3.07, E0 = -2.0, τD = 0.200, U0 = 0.3, τF = 1.5, τS = 0.007)
p0 = ntuple2dict(p0) # make it mutable
z0 = [10.0, 0.982747, 0.367876]

neural_mass = CoupledODEs(neural_mass_rule, z0, p0)

# %% Compute attractors at a range with multstability
E0_multi = -1.75
set_parameter!(neural_mass, :E0, E0_multi)
# Define some arbitrarily large enough grid
density = 21
Eg = range(0, 30; length = density)
xg = range(0, 2; length = density)
ug = range(0, 2; length = density)
grid = (Eg, xg, ug)

# Default mapper
mapper = AttractorsViaRecurrences(
    neural_mass, grid;
    sparse = true,
)

using Random: Xoshiro
sampler, = statespace_sampler(Xoshiro(1234);
    min_bounds = minimum.(grid), max_bounds = maximum.(grid)
)
ics = StateSpaceSet([SVector{3}(sampler()) for _ in 1:1000])

# TODO: We need the "successful step" function here.
# I am not sure what happens when the integration fails
fs, labels, attractors = basins_fractions(mapper, ics)

# Anyways, this found the limit cycle perfectly well
# and trivially fast :)

# %% Basins fractions
E0_range = range(-2, -0.9; length = 51)

rsc = RecurrencesSeededContinuation(mapper)

fractions_curves, attractors_info = continuation(
    rsc, E0_range, :E0, sampler,
)

fig = basins_fractions_plot(fractions_curves, E0_range)

# So fucking easy... ;)