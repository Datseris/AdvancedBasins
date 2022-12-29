using DrWatson
@quickactivate
using Attractors, OrdinaryDiffEq, CairoMakie
using Random
include(srcdir("vis", "basins_plotting.jl"))
include(srcdir("additional_predefined_systems.jl"))
include(srcdir("fractions_produce_or_load.jl"))

# %% Prepare the fractions
fractions_container = []
ylabels = []
attractor_names = []
pranges = []

N = samples_per_parameter = 1000
P = total_parameter_values = 101

# 1. Henon
ds = Systems.henon(; b = 0.3, a = 1.4)
prange = range(1.2, 1.25; length = P)
acritical = 1.2265

xg = yg = range(-2.5, 2.5, length = 500)
pidx = 1
sampler, = statespace_sampler(Random.MersenneTwister(1234);
    min_bounds = [-2,-2], max_bounds = [2,2]
)
# notice that without this special distance function, even with a
# really small threshold like 0.2 we still get a "single" attractor
# throughout the range. Now we get one with period 14, a chaotic,
# and one with period 7 that spans the second half of the parameter range

distance_function = function (A, B)
    # if length of attractors within a factor of 2, then distance is ≤ 1
    return abs(log(2, length(A)) - log(2, length(B)))
end

mapper = AttractorsViaRecurrences(ds, (xg, yg),
    mx_chk_fnd_att = 3000,
    mx_chk_loc_att = 3000
)
continuation = RecurrencesSeedingContinuation(mapper;
    threshold = 0.99, method = distance_function
)
fractions_curves, attractors_info = basins_fractions_continuation(
    continuation, prange, pidx, sampler;
    show_progress = false, samples_per_parameter = N
)

entries = [
 -1 => "diverge",
  1 => "chaotic",
  2 => "period  13",
  3 => "chaotic",
  4 => "period 7",
]
push!(attractor_names, entries)
push!(fractions_container, fractions_curves)
push!(ylabels, "henon")
push!(pranges, prange)


# 2. Population dynamics
ds = competition()
mapper_config = (; Δt= 1.0, mx_chk_fnd_att=9);
xg = range(0, 60; length = 300)
grid = ntuple(x->xg, 8)
pidx = :D
prange = range(0.2, 0.3; length = P)
config = FractionsRecurrencesConfig("populationdynamics", ds, prange, pidx, grid, mapper_config, N)
output = fractions_produce_or_load(config; force = false)
@unpack fractions_curves, attractors_info = output
# Aggregation of attractors based on the existence or not of some unit
unitidxs = 3
featurizer = (A) -> begin
    i = isextinct(A, unitidxs)
    return SVector(Int32(i))
end
isextinct(A, idx = unitidxs) = all(a -> a <= 1e-2, A[:, idx])

# `minneighbors = 1` is crucial for grouping single attractors
groupingconfig = GroupViaClustering(; min_neighbors=1, optimal_radius_method=0.5)

aggregated_fractions, aggregated_info = aggregate_attractor_fractions(
    fractions_curves, attractors_info, featurizer, groupingconfig
)

entries = [1 => "alive", 2 => "extinct"]
push!(attractor_names, entries)
push!(fractions_container, aggregated_fractions)
push!(ylabels, "competition")
push!(pranges, prange)

# 3. Sync basin
# match using the metric we define in the book, the order parameter R,
# the magnitude of the mean field of the frequencies,
# equation (9.8) from our book.
# In the paper we say "in practice, each point in the attractor
# can be considered as one point in time." So you can average
# R over the points of the attractor.

# %% plot
L = length(ylabels)
fig, axs = subplotgrid(L, 1; ylabels)
display(fig)

for i in 1:L
    @show i
    basins_fractions_plot!(axs[i, 1], fractions_container[i], pranges[i])
    # legend
    entries = attractor_names[i]
    if !isnothing(entries)
        elements = [PolyElement(color = COLORS[k]) for k in first.(entries)]
        labels = last.(entries)
        axislegend(axs[i, 1], elements, labels; position = :rt)
    end
end
axs[end, 1].xlabel = "parameter"
rowgap!(fig.layout, 4)
wsave(papersdir("figures", "figure3_matching.png"), fig)
