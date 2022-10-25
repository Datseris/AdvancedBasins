using DrWatson
@quickactivate
using Attractors, DynamicalSystemsBase, OrdinaryDiffEq
using Random
include(srcdir("vis", "basins_plotting.jl"))
diffeq = (alg = Vern9(), reltol = 1e-9, abstol = 1e-9, maxiters = 1e12)

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
g = 201 # division of grid
xgs = [range(-10, 20; length = g÷10) for i in 1:P]
Tg = range(230, 350; length = g)
grid = (xgs..., Tg)

pidx = 1
prange = range(5, 19; length = 100)
prange = [6.0, 12.0, 18.0]
pname = "S"

mapper = AttractorsViaRecurrencesSparse(pinteg, grid;
    Ttr = 100,
    Δt = 1.0,
    mx_chk_fnd_att = 20,
    mx_chk_loc_att = 20,
    safety_counter_max = 1e8,
    diffeq,
)

sampler, = statespace_sampler(Random.MersenneTwister(1234);
    min_bounds = minimum.(grid), max_bounds = maximum.(grid)
)




# %% Normal mapping via `basins_fractions`
S = 18.0
set_parameter!(ds, 1, S)
fracs = basins_fractions(mapper, sampler; N = 100)


# %% Continutation
mapper = AttractorsViaRecurrencesSparse(pinteg, grid;
    Ttr = 100,
    Δt = 1.0,
    mx_chk_fnd_att = 20,
    mx_chk_loc_att = 20,
    safety_counter_max = 1e8,
    diffeq,
)

continuation = RecurrencesSeedingContinuation(mapper; threshold = Inf)
fractions_curves, attractors_info = basins_fractions_continuation(
    continuation, prange, pidx, sampler;
    show_progress = true, samples_per_parameter = 100
)

ukeys = unique_keys(fractions_curves)

# So if we get fractions_curves[80:90]
# we see that just after the transition of 3 to 2 attractors,
# we get  Dict(3 => 0.1400198609731877, 1 => 0.23535253227408143).
# So the fractions do not sum to 1.

# %%
animate_attractors_continuation(
    ds, attractors_info, prange, pidx;
    savename = "test.mp4", access = [1,2],
    limits = (-25,25,-25,25),
)

# Basins plot
# %%

# Barplot of all basins, order by keys
fig = basins_fractions_plot(fractions_curves, prange)

# Makie.save("lorenz84_fracs.png", fig; px_per_unit = 4)
# negate_remove_bg("lorenz84_fracs.png")
