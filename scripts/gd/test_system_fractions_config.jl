using DrWatson
@quickactivate
using Attractors, OrdinaryDiffEq, CairoMakie
using Random
include(srcdir("vis", "basins_plotting.jl"))
include(srcdir("fractions_produce_or_load.jl"))
include(srcdir("predefined_systems.jl"))

N = samples_per_parameter = 10
P = total_parameter_values = 101

# %%

# Climate bistable toy model from Gelbrecht et al. 2021
# using exactly the same projecction as they use in Fig. 2 and also in the MCBB
# algorithm
X = 32 # number of x variables
ds = lorenz96_ebm_gelbrecht_projected(; N = X, S = 18.0)
g = 51 # division of grid

Mg = range(-2, 10; length = g)
Eg = range(0, 50; length = g) # this depends o number of variables
Tg = range(230, 350; length = g)
grid = (Mg, Eg, Tg)
mapper_config = (;
    Ttr = 400,
    Δt = 0.1,
    # We don't care about finding attractors accurately here, because they are
    # so well separated in temperature dimension, and they are only 2.
    # But then again, the worse we identify tje attractor cells, the slower the convergence
    # of new initial conditions will be...?
    mx_chk_fnd_att = 1000,
    mx_chk_loc_att = 1000,
    mx_chk_lost = 500,
    mx_chk_safety = 1e7,
)

pidx = 1
prange = range(5, 19; length = P)

config = FractionsRecurrencesConfig("climatetoy_N=$(X)", ds, prange, pidx, grid, mapper_config, N)

output = fractions_produce_or_load(config; force = false)

@unpack fractions_curves, attractors_info = output

@show unique_keys(fractions_curves)

basins_curves_plot(fractions_curves, prange;
    add_legend = true, separatorwidth = 0
)


# %%

# Climate bistable toy model from Gelbrecht et al. 2021
# Should yield Fig. 3 of the paper
X = 32 # number of x variables
projection_number = 3 # project system to last
ds = lorenz96_ebm_gelbrecht_projected(; N = X, P = projection_number)
g = 21 # division of grid
gT = 101
xgs = [range(-7, 7; length = g) for i in 1:projection_number]
Tg = range(240, 300; length = gT)
grid = (xgs..., Tg)
mapper_config = (;
    Ttr = 400,
    Δt = 0.25,
    # We don't care about finding attractors accurately here, because they are
    # so well separated in temperature dimension, and they are only 2.
    # But then again, the worse we identify tje attractor cells, the slower the convergence
    # of new initial conditions will be...?
    mx_chk_fnd_att = 200,
    mx_chk_loc_att = 100,
    mx_chk_safety = 1e7,
)

pidx = 1
prange = range(5, 19; length = P)

config = FractionsRecurrencesConfig("climatetoy_N=$(X)", ds, prange, pidx, grid, mapper_config, N)

output = fractions_produce_or_load(config; force = false)

@unpack fractions_curves, attractors_info = output

basins_curves_plot(fractions_curves, prange;
    add_legend = false, separatorwidth = 0
)