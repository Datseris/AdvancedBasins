using DrWatson
@quickactivate
using Attractors
using LinearAlgebra: norm
include(srcdir("vis", "basins_plotting.jl"))

# %%
d, α, ω = 0.3, 0.2, 0.5
ds = Systems.magnetic_pendulum(; d, α, ω)
xg = yg = range(-3, 3, length = 101)
ds = projected_integrator(ds, 1:2, [0.0, 0.0])
mapper = AttractorsViaRecurrences(ds, (xg, yg); Δt = 1.0)
rr = range(1, 0.2; length = 201)
ps = [[1, 1, γ] for γ in rr]
pidx = :γs
# test that both finding and removing attractor works
mapper = AttractorsViaRecurrences(ds, (xg, yg); Δt = 1.0)
sampler, = statespace_sampler(; min_bounds = [-3,-3], max_bounds = [3,3])
continuation = RecurrencesSeedingContinuation(mapper; threshold = Inf)
# With this threshold all attractors are mapped to each other, they are within
# distance 1 in state space.
fractions_curves, attractors_info = basins_fractions_continuation(
    continuation, ps, pidx, sampler; show_progress = true, samples_per_parameter = 1000
)

# %% Special plots
fig = Figure()
axbasins = [Axis(fig[1,i]) for i in 1:3]
axfrac = Axis(fig[2,:])
prange = [p[3] for p in ps]
basins_fractions_plot!(axfrac, reverse(fractions_curves), reverse(prange))
resize_to_layout!(fig)
fig

# add fractions
xg = yg = range(-3, 3, length = 401)

chosenps = [[1, 1, 1.0], [1, 1, 0.5], [1, 1, 0.2]]

for (i, p) in enumerate(chosenps)
    set_parameter!(ds, pidx, p)
    mapper = AttractorsViaRecurrences(ds, (xg, yg); Δt = 1.0)
    b, att = basins_of_attraction(mapper)

    closest_j = findfirst(isequal(p), ps)
    @show closest_j
    rmap = replacement_map(att, attractors_info[closest_j])
    @show rmap
    replace!(b, pairs(rmap)...)

    heatmap!(axbasins[i], xg, yg, b; colormap = categorical_colors(COLORSCHEME, 3))
end

hideydecorations!(axfrac)
axfrac.xlabel = L"\gamma"
axfrac.xreversed = true
axfrac.xticks = range(0, 1; length = 6)
for i in 1:3
    hidedecorations!(axbasins[i])
end


import CairoMakie
wsave(plotsdir("presentations", "magnetic"), fig)