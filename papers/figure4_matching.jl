using DrWatson
@quickactivate
using DynamicalSystems, OrdinaryDiffEq, CairoMakie
using GLMakie, Random
include(srcdir("vis", "basins_plotting.jl"))

# 1. Henon
ds = Systems.henon(; b = 0.3, a = 1.4)
psorig = range(1.2, 1.25; length = 101)
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
    threshold = 0.99, metric = distance_function
)
ps = psorig
fractions_curves, attractors_info = basins_fractions_continuation(
    continuation, ps, pidx, sampler;
    show_progress = false, samples_per_parameter = 10
)


# 2. Sync basin
# match using the metric we define in the book, the order parameter R,
# the magnitude of the mean field of the frequencies,
# equation (9.8) from our book.
# In the paper we say "in practice, each point in the attractor
# can be considered as one point in time." So you can average
# R over the points of the attractor.
