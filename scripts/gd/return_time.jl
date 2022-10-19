using DrWatson
@quickactivate
using DynamicalSystems
using OrdinaryDiffEq, GLMakie

# %% Step 1, produce the results, using the `produce_or_load`
D = 6
ds = Systems.lorenz96(D; F = 32.0)
diffeq = (alg = Vern9(), abstol = 1e-9, reltol = 1e-9, maxiters = typemax(Int))
tr = trajectory(ds, 10; diffeq)

# For each D we integrate for a random amount of time,
# then we pick x states, and then for these radii we calculate return times
# Practically for any D the ranges of variables are ±40
εs = 40 ./ (2 .^ [5, 6, 7, 8])
threshold_multiplier = 100

tr = trajectory(ds, 100 + 100rand(); diffeq, Ttr = 10000)
x, y, z = columns(tr)[[1, 4, 6]]

# fig, ax = lines(x,y,z)
# display(fig)

u0 = tr[end]
T = 1e7
τ = first_return_times(ds, u0, εs, T; diffeq, crossing_method = CrossingAccurateInterpolation())
# τ, c = exit_entry_times(ds, u0, εs, T; diffeq, threshold_multiplier)