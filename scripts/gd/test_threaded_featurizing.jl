using DrWatson
@quickactivate
using DynamicalSystems
using OrdinaryDiffEq
using BenchmarkTools
using Statistics
using Graphs
using Random

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
P = [isodd(i) ? +1. : -1. for i = 1:D]
K = 1.0

ds = ContinuousDynamicalSystem(second_order_kuramoto!, zeros(2*D), [D, 0.1, K, E, vec(P)], (J, z0, p, n) -> nothing)
diffeq = (alg = Tsit5(), reltol = 1e-9, maxiters = 1e6)

function featurizer(A, t)
    return [std(A[:, i]) for i in 1:5]
end

mapper_single = AttractorsViaFeaturizing(ds, featurizer; diffeq)
mapper_thread = AttractorsViaFeaturizing(ds, featurizer; diffeq, threaded = true)

N = 2000
ics = SVector{2D, Float64}[]
for k = 1:N
    u = vec([pi.*rand(D); (rand(D) .- 0.5).*12])
    push!(ics, u)
end
ics = Dataset(ics)


ChaosTools.extract_features(mapper_single, ics)
ChaosTools.extract_features(mapper_thread, ics)

# @btime ChaosTools.extract_featres(mapper, ics)
# @btime ChaosTools.extract_featres(mapper, )