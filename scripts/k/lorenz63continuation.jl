using DrWatson
@quickactivate
using DynamicalSystems, Random, CairoMakie

ds = Systems.lorenz(ρ=26)
ps = [
    0.5, #stable origin
    1.15, #two stable fps
    1.5, #two stale foci
    5, #still two stable foci
    22.0, #two stable foci and a chaotic saddle
    24.0, #still
    24.1, #two stable foci and a chaotic attractor
    25, #chaotic attractor
    28 #chaotic attractor
]
ps_description = [
    "stable origin",
    "two stable fps",
    "two stable foci",
    "two stable foci",
    "two stable foci + chaotic saddle",
    "two stable foci + chaotic saddle",
    "two stable foci + chaotic att",
    "chaotic attractor",
    "chaotic attractor",
]

# param_mode = "movingforwards"
param_mode = "movingbackwards"
if param_mode == "movingbackwards" ps = ps[end:-1:1]; ps_description = ps_description[end:-1:1]; end

xg = yg = zg = range(-60, 60, length = 500)
mapper = AttractorsViaRecurrences(ds, (xg, yg, zg),
mx_chk_fnd_att = 1000,
mx_chk_loc_att = 1000,
mx_chk_att = 100,
mx_chk_lost = 1000,
safety_counter_max = Int(1e9),
sparse=true
)
pidx = 2
sampler, = statespace_sampler(Random.MersenneTwister(1234);
    min_bounds = [-40,-40,-40], max_bounds = [40, 40, 40]
)

continuation = RecurrencesSeedingContinuation(mapper)
fractions_curves, attractors_info = basins_fractions_continuation(
    continuation, ps, pidx, sampler;
    show_progress = false, samples_per_parameter = 500
)

colors = Colors.distinguishable_colors(maximum(keys(fractions_curves)), [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

fractions_curves
idxs = CartesianIndices(reshape(1:length(ps), (3,3))')
fig = Figure(resolution=(1000, 1000))
for (i, idx) ∈ enumerate(idxs)
    ax = Axis(fig[idx[1], idx[2]], title="ρ = $(ps[i])\n $(ps_description[i])", xlabel="x", ylabel="y")
    limits!(ax, -30, 30, -30, 30)
    for (key, att) in attractors_info[i]
        mk = length(att[:,1]) > 10 ? 1.0 : 10
        scatter!(ax, att[:,1], att[:,2], markersize=mk, color=colors[key])
    end
end
supertitle = Label(fig[0, :], "Results from continuation in Lorenz63, ρ $param_mode \n each subtitle describes all structures, whether found or not", textsize = 20)
fig
save("$(plotsdir())/k/continuationtest/lorenz63-$param_mode.png", fig, px_per_unit=4)