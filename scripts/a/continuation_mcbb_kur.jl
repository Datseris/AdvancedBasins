using DrWatson
@quickactivate
using Revise
using DynamicalSystems
using Attractors
using Random
using Graphs
using OrdinaryDiffEq:Tsit5
using Statistics:mean

function second_order_kuramoto!(du, u, p, t)
    (; N, α, K, incidence, P) = p
    ωs = view(u, N+1:2N)
    du[1:N] .= ωs
    sine_term = K .* (incidence * sin.(incidence' * u[1:N]))
    @. du[N+1:end] .= P - α*ωs - sine_term
    return nothing
end

mutable struct KuramotoParameters{M}
    N::Int
    α::Float64
    incidence::M
    P::Vector{Float64}
    K::Float64
end
function KuramotoParameters(; N = 10, α = 0.1, K = 6.0, seed = 53867481290)
    rng = Random.Xoshiro(seed)
    g = random_regular_graph(N, 3; rng)
    incidence = incidence_matrix(g, oriented=true)
    P = [isodd(i) ? +1.0 : -1.0 for i = 1:N]
    return KuramotoParameters(N, α, incidence, P, K)
end

function continuation_problem()
    N = 30
    p = KuramotoParameters(; N)
    ds = ContinuousDynamicalSystem(second_order_kuramoto!, zeros(2*N), p, (J, z0, p, n) -> nothing)
    diffeq = (alg = Tsit5(), reltol = 1e-9, maxiters = 1e6)

    function featurizer(A, t)
        return [mean(A[:, i]) for i in N+1:2*N]
    end

    clusterspecs = Attractors.GroupViaClustering(optimal_radius_method = "silhouettes", max_used_features = 500)
    mapper = Attractors.AttractorsViaFeaturizing(ds, featurizer, clusterspecs; T = 200, Ttr = 400, diffeq)

    sampler, = ChaosTools.statespace_sampler(Random.MersenneTwister(1234);
        min_bounds = [-pi*ones(N) -pi*ones(N)], max_bounds = [pi*ones(N) pi*ones(N)]
    )

    continuation = ClusteringAcrossParametersContinuation(mapper; use_mmap = true)
    Kidx = :K
    Krange = range(0., 10; length = 20)
    f, a = basins_fractions_continuation(
                continuation, Krange, Kidx, sampler;
                show_progress = true, samples_per_parameter = 5000)
    return f, a, Krange
end

f,a,Krange = continuation_problem()
save("fractions_cont_mccb_kur.jld2", "f", f, "a", a, "K", Krange)
