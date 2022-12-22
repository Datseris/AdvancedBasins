using DrWatson
@quickactivate
using Attractors, OrdinaryDiffEq, CairoMakie
using Random

include(srcdir("vis", "basins_plotting.jl"))
include(srcdir("fractions_produce_or_load.jl"))
include(srcdir("additional_predefined_systems.jl"))

N = samples_per_parameter = 1000
P = total_parameter_values = 101
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

basins_fractions_plot!(axs[1,1], fractions_curves, prange; separatorwidth = 1)

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

basins_fractions_plot!(axs[2,1], aggregated_fractions, prange; separatorwidth = 1)

wsave(plotsdir("gd", "competition_dynamics_aggregation.png"), fig)

# %% same plot, special colors
fig, axs = subplotgrid(2,1)
axs[1].title = "original"
axs[2].title = "grouped"
display(fig)

extinct_color = colorant"red"
survive_color = colorant"green"

ukeys = unique_keys(attractors_info)
label_extincts = map(atts->[k for (k,v) in atts if isextinct(v)], attractors_info)
label_extincts = unique!(reduce(vcat, label_extincts))
label_survive = [key for key in ukeys if key ∉ label_extincts]
labels = [k in label_extincts ? "extinct" : "survive" for k in ukeys]

using Makie.Colors
colors_survive = length(label_survive) == 1 ? [survive_color] : collect(range(colorant"darkolivegreen2", stop=survive_color, length=length(label_survive)))
colors_extinct   = length(label_extincts) == 1 ? [extinct_color] : collect(range(extinct_color, stop=colorant"red4", length=length(label_extincts)))
colors = merge(Dict(label_survive .=> colors_survive), Dict(label_extincts .=> colors_extinct))

basins_fractions_plot!(axs[1,1], fractions_curves, prange;
colors = colors, labels, add_legend = false, separatorwidth = 1)

# now for the grouped
labels_grouped = [only(v) < 0.5 ? "extinct" : "survive" for v in values(aggregated_info)]
colors_grouped = [l == "extinct" ? extinct_color : survive_color for l in labels_grouped]

basins_fractions_plot!(axs[2,1], aggregated_fractions, prange;
colors = colors_grouped, labels=labels_grouped, separatorwidth = 1)

wsave(plotsdir("gd", "competition_dynamics_aggregation_specialcolor.png"), fig)
