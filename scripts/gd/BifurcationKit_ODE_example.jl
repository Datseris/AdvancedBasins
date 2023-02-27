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
z0 = [0.238616, 0.982747, 0.367876]

neural_mass = CoupledODEs(neural_mass_rule, z0, p0)
