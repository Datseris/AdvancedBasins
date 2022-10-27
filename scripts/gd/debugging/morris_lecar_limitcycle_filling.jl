using DrWatson
@quickactivate
using Attractors
using GLMakie, Random, Statistics
using OrdinaryDiffEq: Tsit5, Vern9
using ChaosTools: estimate_period

include(srcdir("vis", "basins_plotting.jl"))

stop_at_dt = true
store_once_per_cell = true
Δt = 0.15
mx_chk_loc_att = 1000
mx_chk_fnd_att = 2
I = 0.115

ds = Systems.morris_lecar(; I)
diffeq = (reltol = 1e-9, abstol = 1e-9, alg = Vern9())
xg = yg = range(-1,1,length = 20000)
mapper = AttractorsViaRecurrences(ds, (xg, yg);
    mx_chk_fnd_att,
    mx_chk_loc_att,
    stop_at_dt,
    store_once_per_cell,
    sparse = true,
    Δt,
    Ttr = 10,
)

sampler, = Attractors.statespace_sampler(Random.MersenneTwister(1);
    min_bounds = [-0.5, 0], max_bounds = [0.5, 1])
ics = Dataset([sampler() for i in 1:1000])

mean_integ_dt = let
    integ = integrator(ds, rand(vec(ics)))
    step!(integ, 100)
    times = zeros(1000)
    for i in 1:1000
        step!(integ)
        times[i] = integ.dt
    end
    mean(times)
end

period = let
    tr = trajectory(ds, 10000.0, rand(vec(ics)); Δt = 0.05, Ttr = 100.0)
    p = estimate_period(tr[:, 1], :autocorrelation; ϵ = 0.1)
    p*0.05
end

fs, labels, atts = basins_fractions(mapper, ics; show_progress=false)

fig = plot_attractors(atts)
display(fig)
ax = content(fig[1,1])
label = """
I = $I
mx_chk_loc_att = $(mx_chk_loc_att)
mx_chk_fnd_att = $(mx_chk_fnd_att)
stop_at_dt = $(stop_at_dt)
store_once_per_cell = $(store_once_per_cell)
mean_integ_dt = $(round(mean_integ_dt; sigdigits = 3))
period = $(period)
Δt = $(Δt)
"""

Label(fig[1,1], label;
    tellwidth=false, tellheight=false,
    valign = :top, halign = :left, justification = :left
)