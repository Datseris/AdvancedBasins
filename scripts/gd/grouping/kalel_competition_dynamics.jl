using DrWatson
@quickactivate
using Attractors, OrdinaryDiffEq, CairoMakie
using Random
include(srcdir("vis", "basins_plotting.jl"))
include(srcdir("fractions_produce_or_load.jl"))
include(srcdir("additional_predefined_systems.jl"))

N = samples_per_parameter = 300
P = total_parameter_values = 21
fig, axs = subplotgrid(2,1)
axs[1].title = "original"
axs[2].title = "grouped"
display(fig)

# Population dynamics recurrences continuation. It uses the DrWatson integration
# pipeline to not re-compute things. But all in all, it is a direct call to
# `RecurrencesSeededContinuation` with default matching behavior
# (distance in state space). **No special matching or metric is used here!!!**
ds = competition()
mapper_config = (; Δt= 1.0, mx_chk_fnd_att=9);
xg = range(0, 60; length = 300);
grid = ntuple(x->xg, 8);
pidx = :D
prange = range(0.2, 0.3; length = P)
config = FractionsRecurrencesConfig("populationdynamics", ds, prange, pidx, grid, mapper_config, N)

output = fractions_produce_or_load(config; force = false)

@unpack fractions_curves, attractors_info = output

basins_fractions_plot!(axs[1,1], fractions_curves, prange)

# Aggregation of attractors based on the existence or not of some unit
unitidxs = 3

featurizer = (A) -> begin
    i = isextinct(A, unitidxs)
    return SVector(Int32(i))
end
isextinct(A, idx) = all(a -> a <= 1e-2, A[:, idx])

# `minneighbors = 1` is crucial for grouping single attractors
groupingconfig = GroupViaClustering(; min_neighbors=1, optimal_radius_method=0.5)

joint_fractions = aggregate_attractor_fractions(
    fractions_curves, attractors_info, featurizer, groupingconfig
)

basins_fractions_plot!(axs[2,1], joint_fractions, prange)

wsave(plotsdir("gd", "competition_dynamics_aggregation.pdf"), fig)
