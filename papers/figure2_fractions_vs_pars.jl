using DrWatson
@quickactivate
using Attractors, OrdinaryDiffEq, CairoMakie
using GLMakie, Random
include(srcdir("vis", "basins_plotting.jl"))

# Global arguments for all sub-panels
N = samples_per_parameter = 100
R = total_parameter_values = 101
diffeq = (alg = Vern9(), reltol = 1e-9, abstol = 1e-9, maxiters = 1e12)

# In `systems_param_configs` we put a tuple of arguments related to the system,
# such as the  name, system, parameter range and index,
# and the mapper configuration options (they are expanded to mapper later).
# We are assuming that the system parameters and their grid won't be changing
# in any way, so they do not take part in the `produce_or_load` decision-making.
# (The grid resolution however does take part in the naming!)

systems_param_configs = []

# Lorenz84
F = 6.886; G = 1.347; a = 0.255; b = 4.0
ds = Systems.lorenz84(; F, G, a, b)
M = 600; z = 3
xg = yg = zg = range(-z, z; length = M)
grid = (xg, yg, zg)
mapper_config = (;
    mx_chk_fnd_att = 5000,
    mx_chk_loc_att = 50000,
    mx_chk_att = 2,
    mx_chk_lost = 1000,
    safety_counter_max = 1e8,
    Δt = 0.1,
)
prange = range(1.34, 1.37; length = R)
pidx = 2
pname = "G"
entries = [1 => "f.p.", 2 => "l.c.", 3 => "c.a."]
push!(systems_param_configs, ("lorenz84", ds, prange, pidx, pname, grid, mapper_config, entries))

# Lorenz63
ds = Systems.lorenz()
M = 200
xg = yg = range(-25, 25; length = M)
zg = range(0, 60; length = M)
grid = (xg, yg, zg)
mapper_config = (;
    mx_chk_fnd_att = 2000,
    mx_chk_loc_att = 500,
    mx_chk_att = 2,
    Ttr = 2000,
    safety_counter_max = 1e8,
    Δt = 0.1,
)
prange = range(22.0, 26.0; length = R)
pname = "ρ"
pidx = 2
entries = [1 => "f.p.", 2 => "f.p.", 3 => "c.a."]
push!(systems_param_configs, ("lorenz63", ds, prange, pidx, pname, grid, mapper_config, entries))

# Climate bistable toy model from Gelbrecht et al. 2021
# Should yield Fig. 3 of the paper
function lorenz96_ebm_gelbrecht(dx, x, p, t)
    N = length(x) - 1 # number of grid points of Lorenz 96
    T = x[end]
    a₀ = 0.5
    a₁ = 0.4
    S = p[1] # Solar constant, by default 8.0
    F = 8.0
    Tbar = 270.0
    ΔT = 60.0
    α = 2.0
    β = 1.0
    σ = 1/180
    E = 0.5*sum(x[n]^2 for n in 1:N)
    𝓔 = E/N
    forcing = F*(1 + β*(T - Tbar)/ΔT)
    # 3 edge cases
    @inbounds dx[1] = (x[2] - x[N - 1]) * x[N] - x[1] + forcing
    @inbounds dx[2] = (x[3] - x[N]) * x[1] - x[2] + forcing
    @inbounds dx[N] = (x[1] - x[N - 2]) * x[N - 1] - x[N] + forcing
    # then the general case
    for n in 3:(N - 1)
      @inbounds dx[n] = (x[n + 1] - x[n - 2]) * x[n - 1] - x[n] + forcing
    end
    # Temperature equation
    dx[end] = S*(1 - a₀ + (a₁/2)*tanh(T-Tbar)) - (σ*T)^4 - α*(𝓔/(0.6*F^(4/3)) - 1)
    return nothing
end
# Initial parameters
p0 = [8.0] # solar constant
D = 32 # number of X variables
ds = ContinuousDynamicalSystem(lorenz96_ebm_gelbrecht, [rand(D)..., 230.0], p0)
# Project system
P = 6 # project system to last `P` variables
projection = (D-P+1):(D+1)
complete_state = zeros(D-length(projection)+1)
pinteg = projected_integrator(ds, projection, complete_state)
# Make grid
g = 101 # division of grid
xgs = [range(-8, 15; length = g÷10) for i in 1:P]
Tg = range(230, 350; length = g)
grid = (xgs..., Tg)

# Push stuff into fraction continuation configurations
mapper_config = (;
    Ttr = 100,
    Δt = 1.0,
    mx_chk_fnd_att = 20,
    mx_chk_loc_att = 20,
    safety_counter_max = 1e8,
)
pidx = 1
prange = range(5, 19; length = R)
pname = "S"
entries = [1 => "cold", 2 => "warm"]
push!(systems_param_configs, ("climatetoy_N=$(D)", pinteg, prange, pidx, pname, grid, mapper_config, entries))


# produce or load function
function produce_basins_fractions(config)
    (; ds, prange, pidx, grid, mapper_config) = config
    mapper = AttractorsViaRecurrencesSparse(ds, grid; mapper_config..., diffeq)
    sampler, = statespace_sampler(Random.MersenneTwister(1234);
        min_bounds = minimum.(grid), max_bounds = maximum.(grid)
    )
    continuation = RecurrencesSeedingContinuation(mapper; threshold = Inf)
    fractions_curves, attractors_info = basins_fractions_continuation(
        continuation, prange, pidx, sampler;
        show_progress = true, samples_per_parameter = N
    )
    return @strdict(fractions_curves, attractors_info)
end


fractions_container = []
for entry in systems_param_configs
    name, ds, prange, pidx, pname, grid, mapper_config = entry
    G = prod(length.(grid))
    config_for_name = (; N, G, mapper_config...)
    filename = savename(name, config_for_name, "jld2")
    config_for_produce = (; ds, prange, pidx, grid, mapper_config)
    # call the producing function
    output, file = produce_or_load(
        produce_basins_fractions, config_for_produce, datadir("basins_fractions");
        filename, force = false,
    )
    push!(fractions_container, output["fractions_curves"])
end

systems = getindex.(systems_param_configs, 1)

# %% Make the plot
L = length(systems_param_configs)
fig, axs = subplotgrid(L, 1; ylabels = systems)

for i in 1:L
    prange = systems_param_configs[i][3]
    basins_fractions_plot!(axs[i, 1], fractions_container[i], prange)
    # legend
    entries = systems_param_configs[i][end]
    elements = [PolyElement(color = COLORS[k]) for k in first.(entries)]
    labels = last.(entries)
    axislegend(axs[i, 1], elements, labels; position = :rt)
end
axs[end, 1].xlabel = "parameter"
fig

# wsave(papersdir("figures", "figure2_fractions.pdf"), fig)