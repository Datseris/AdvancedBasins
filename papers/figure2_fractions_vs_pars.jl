# %% Setup
# Create figure 2 of main text: Fractions of attractors versus parameters.
# Script heavily relies on the DrWatson integration with `produce_or_load`.
# See included `src` files below.
using DrWatson
@quickactivate
using Attractors, OrdinaryDiffEq, CairoMakie
# using GLMakie, Random
using Random
include(srcdir("vis", "basins_plotting.jl"))
include(srcdir("fractions_produce_or_load.jl"))
include(srcdir("predefined_systems.jl"))

# Global arguments for all sub-panels

N = samples_per_parameter = 1000
P = total_parameter_values = 101

# N = samples_per_parameter = 100
# P = total_parameter_values = 51

# N = samples_per_parameter = 10
# P = total_parameter_values = 11

# %% Used systems
# In `configs` we put an instance of `FractionsRecurrencesConfiguration`
# for each system. In the other containers we put things related to plotting if need be.
configs = []
attractor_names = []

# Lorenz84
F = 6.886; G = 1.347; a = 0.255; b = 4.0
ds = lorenz84(; F, G, a, b)
M = 600; z = 3
xg = yg = zg = range(-z, z; length = M)
grid = (xg, yg, zg)
mapper_config = (;
    mx_chk_fnd_att = 1000,
    mx_chk_loc_att = 2000,
    mx_chk_att = 4,
    mx_chk_lost = 100,
    mx_chk_safety = 1e8,
    Ttr = 10,
    Δt = 0.05,
    force_non_adaptive = true,
)
prange = range(1.34, 1.37; length = P)
pidx = 2
entries = [1 => "f.p.", 2 => "l.c.", 3 => "c.a."]
config = FractionsRecurrencesConfig("lorenz84", ds, prange, pidx, grid, mapper_config, N)
push!(configs, config)
push!(attractor_names, entries)

# Climate bistable toy model from Gelbrecht et al. 2021
# Should yield Fig. 3 of the paper
X = 16 # number of x variables
projection_number = 6 # project system to last
ds = lorenz96_ebm_gelbrecht_projected(; N = X, P = projection_number)
g = 101 # division of grid
xgs = [range(-8, 15; length = g÷10) for i in 1:projection_number]
Tg = range(230, 350; length = g)
grid = (xgs..., Tg)
mapper_config = (;
    Ttr = 500,
    Δt = 0.25,
    # We don't care about finding attractors accurately here, because they are
    # so well separated in temperature dimension, and they are only 2.
    # But then again, the worse we identify tje attractor cells, the slower the convergence
    # of new initial conditions will be...?
    mx_chk_fnd_att = 10,
    mx_chk_loc_att = 10,
    mx_chk_safety = 1e6,
)
pidx = 1
prange = range(5, 19; length = P)
entries = [1 => "cold", 2 => "warm"]
config = FractionsRecurrencesConfig("climatetoy_N=$(X)", ds, prange, pidx, grid, mapper_config, N)
push!(configs, config)
push!(attractor_names, entries)

# Cell differentiation model
ds = cell_differentiation()
mapper_config = (;mx_chk_safety = Int(1e9))
grid = ntuple(i -> range(0, 100, length=101), 3)
pidx = 1 # parameter Kd
prange = range(1e-2, 1e2; length = P)
entries = nothing
config = FractionsRecurrencesConfig("cells", ds, prange, pidx, grid, mapper_config, N)
push!(configs, config)
push!(attractor_names, entries)

# Eckhardt 9D sheer flow model
ds = Eckhardt_9D()
yg = range(-2, 2; length = 1001)
grid = ntuple(x -> yg, 9)
mapper_config = (; sparse = true, Δt = 1.0,
    mx_chk_fnd_att = 2500,
    force_non_adaptive = true, 
    store_once_per_cell = true,
    mx_chk_loc_att = 2500, mx_chk_safety = Int(1e7), show_progress = true,
    mx_chk_att = 10
)
pidx = :Re
sampler, = Attractors.statespace_sampler(Random.MersenneTwister(1234); min_bounds = ones(9).*(-1.), max_bounds = ones(9).*(1.))
prange = range(300, 450; length = 25)
entries = [1 => "Laminar", 3 => "Turbulent 1", 4 => "Turbulent 2"]
config = FractionsRecurrencesConfig("eckhardt", ds, prange, pidx, grid, mapper_config, N, Inf, sampler)
push!(configs, config)
push!(attractor_names, entries)

# Population dynamics
ds = competition()
mapper_config = (; Δt = 1.0, mx_chk_fnd_att=9);
xg = range(0, 60; length = 300); grid = ntuple(x->xg, 8);
pidx = :D
entries = nothing
prange = range(0.2, 0.3; length = P)
config = FractionsRecurrencesConfig("populationdynamics", ds, prange, pidx, grid, mapper_config, N)
push!(configs, config)
push!(attractor_names, entries)

# %% Run all systems through the `produce_or_load` pipeline (see `src`)
fractions_container = []
for config in configs
    output = fractions_produce_or_load(config; force = false)
    push!(fractions_container, output["fractions_curves"])
end

# %% Make the plot
systems = getproperty.(configs, :name)
systems = [split(s, '_')[1] for s in systems]
systems[1] = "paradigmatic chaotic\nmodel (lorenz84)"
systems[2] = "high-dim. climate\ntoy model"
systems[3] = "cell\ndifferentiation"
systems[4] = "turbulent\nflow"
systems[5] = "ecological\ncompetition\ndynamics"
L = length(configs)
fig, axs = subplotgrid(L, 1; ylabels = systems, resolution = (1000, 800),)

for i in 1:L
    prange = configs[i].prange
    basins_curves_plot!(axs[i, 1], fractions_container[i], prange;
        add_legend = false, separatorwidth = 0
    )
    # legend
    @show entries = attractor_names[i]
    if !isnothing(entries)
        elements = [PolyElement(color = COLORS[k]) for k in first.(entries)]
        labels = last.(entries)
        axislegend(axs[i, 1], elements, labels; position = :rt)
    end
end
axs[end, 1].xlabel = "parameter"
rowgap!(fig.layout, 4)
display(fig)

# %% Save it
wsave(papersdir("figures", "figure2_fractions.png"), fig)
