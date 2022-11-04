#for multifate model
using DrWatson
@quickactivate
using Revise
using DynamicalSystems, OrdinaryDiffEq

@inbounds function multifate!(du, u, p, t)
    Kd, α, β, n = p
    sum_u = sum(u)
    for i ∈ eachindex(du)
        C = (2*u[i]^2) / (Kd + 4*sum_u + sqrt( Kd^2 + 8*sum_u*Kd )  )
        du[i] = α + (β*C^n)/(1+C^n) - u[i]
    end
end

N = 3; Kd = 1; n = 1.5; α = 0.8; β = 20 #3D example
# N = 2; Kd = 1; n = 1; α = 0.2; β = 15 #2d type II tristability
p = [Kd, α, β, n]


# ---------------------------------------------------------------------------- #
#                            Continuation                                      #
# ---------------------------------------------------------------------------- #
using Random
using Attractors, DelayEmbeddings
N = 3;
α=4; β = 20; n = 1.5; Kds = range(1e-2, 1e2, length=50); p = [Kds[1], α, β, n];
ds = ContinuousDynamicalSystem(multifate!, rand(N), p);
xg = yg = zg = range(0, 100, length=100);
grid = ntuple(x->xg, N);

mapper = Attractors.AttractorsViaRecurrences(ds, grid;  diffeq=(alg=Vern9(),), sparse=true,
mx_chk_safety = Int(1e9));
sampler, = statespace_sampler(Random.MersenneTwister(1234);
    min_bounds = [0.0, 0, 0], max_bounds = [50, 50, 50]
);

continuation = Attractors.RecurrencesSeedingContinuation(mapper);

pidx = 1; #vary Kd
fractions_curves, attractors_info = basins_fractions_continuation(
    continuation, Kds, pidx, sampler;
    show_progress = false, samples_per_parameter = 1000
);

fractions_curves
num_fps = length.(values(fractions_curves))

# using JLD2
# jldsave("$(datadir())/fractions_curves/fractions_curves_celldifferentiation.jld2"; fractions_curves, params=Kds)

basins_fractions_plot(fractions_curves, Kds)

# ---------------------------------------------------------------------------- #
#                            Replicating the results                           #
# ---------------------------------------------------------------------------- #

ds = ContinuousDynamicalSystem(multifate!, rand(N), p)
# tr = trajectory(ds, 10000.0, u0)

xg = yg = zg = range(0, 35, length=100);
grid = ntuple(x->xg, N)
mapper = AttractorsViaRecurrences(ds, grid;  diffeq=(alg=Vern9(),), sparse=true);
basins, attractors = basins_of_attraction(mapper, grid)

using CairoMakie
fig = Figure()
if N == 2
ax = Axis(fig[1,1], xlabel="A", ylabel="B")
heatmap!(ax, xg, yg, basins)
for i=1:length(attractors) scatter!(ax, Matrix(attractors[i])[:,1], Matrix(attractors[i])[:,2], color=:black, markersize=10) end
else
    ax = Axis3(fig[1,1], xlabel="A", ylabel="B")
    for i=1:length(attractors) scatter!(ax, Matrix(attractors[i])[:,1], Matrix(attractors[i])[:,2], Matrix(attractors[i])[:,3], color=:black, markersize=10) end
end
display(fig)

# save("$(plotsdir())/k/celldifferentiation/multifate-N_$N-Kd_$Kd-n_$n-α_$α-β_$β.png", fig)

# ----------------- Conclusion: can replicate their results! ----------------- #
