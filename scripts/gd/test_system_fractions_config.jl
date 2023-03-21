using DrWatson
@quickactivate
using Attractors, OrdinaryDiffEq, CairoMakie
using Random
include(srcdir("vis", "basins_plotting.jl"))
include(srcdir("fractions_produce_or_load.jl"))
include(srcdir("predefined_systems.jl"))

N = samples_per_parameter = 1000
P = total_parameter_values = 101

# Climate bistable toy model from Gelbrecht et al. 2021
# Should yield Fig. 3 of the paper
X = 16 # number of x variables
projection_number = 5 # project system to last P+1 dimensions
ds = lorenz96_ebm_gelbrecht_projected(; N = X, P = projection_number)
g = 101 # division of grid
xgs = [range(-8, 15; length = g÷10) for i in 1:projection_number]
Tg = range(230, 350; length = g)
grid = (xgs..., Tg)
mapper_config = (;
    Ttr = 100,
    Δt = 0.5,
    mx_chk_fnd_att = 1000,
    mx_chk_loc_att = 2000,
    mx_chk_att = 4,
    mx_chk_safety = 1e7,
)
pidx = 1
prange = range(5, 19; length = P)

config = FractionsRecurrencesConfig("climatetoy_N=$(X)", ds, prange, pidx, grid, mapper_config, N)

output = fractions_produce_or_load(config; force = false)

@unpack fractions_curves, attractors_info = output

basins_curves_plot(fractions_curves, prange;
    add_legend = false, separatorwidth = 0e
)