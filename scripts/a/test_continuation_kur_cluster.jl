using DrWatson
@quickactivate
using Revise
using DynamicalSystems
using BenchmarkTools
using Random
using DataFrames
using Graphs
using OrdinaryDiffEq:Tsit5


function second_order_kuramoto!(du, u, p, t)
    D = p[1]; α = p[2]; K = p[3]; incidence = p[4]; P = p[5];
    du[1:D] .= u[1+D:2*D]
    du[D+1:end] .= P .- α .* u[1+D:2*D] .- K .* (incidence * sin.(incidence' * u[1:D]))
end

seed = 5386748129040267798
Random.seed!(seed)
# Set up the parameters for the network
D = 10 # in this case this is the number of oscillators, the system dimension is twice this value
g = random_regular_graph(D, 3)
E = incidence_matrix(g, oriented=true)
P = [isodd(i) ? +1.0 : -1.0 for i = 1:D]
K = 5.0

ds = ContinuousDynamicalSystem(second_order_kuramoto!, zeros(2*D), [D, 0.1, K, E, vec(P)], (J, z0, p, n) -> nothing)
diffeq = (alg = Tsit5(), reltol = 1e-9, maxiters = 1e6)

function featurizer(A, t)
    return [mean(A[:, i]) for i in D+1:2*D]
end

clusterspecs = ClusteringConfig()
mapper = AttractorsViaFeaturizing(ds, featurizer, clusterspecs; T = 200, Ttr = 400, diffeq)

sampler, = statespace_sampler(Random.MersenneTwister(1234);
    min_bounds = [-pi*ones(D) -12*ones(D)], max_bounds = [pi*ones(D) 12*ones(D)]
)

continuation = ClusteringAcrossParametersContinuation(mapper)
Kidx = 3
Krange = range(0., 10; length = 10)
fractions_curves, attractors_info = basins_fractions_continuation(
continuation, Krange, Kidx, sampler;
show_progress = true, samples_per_parameter = 1000)

save("fraction_test_continuation_kur.jld2", "f", fractions_curves)
