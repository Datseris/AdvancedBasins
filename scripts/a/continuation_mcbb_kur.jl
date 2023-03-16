using DrWatson
# @quickactivate
using Revise
# using DynamicalSystems
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

function continuation_problem(di)
    @unpack Nd, Ns = di
    N = Nd
    p = KuramotoParameters(; N)
    diffeq = (alg = Tsit5(), reltol = 1e-9, maxiters = 1e6)
    ds = CoupledODEs(second_order_kuramoto!, zeros(2*N), p; diffeq)

    function featurizer(A, t)
        return [mean(A[:, i]) for i in N+1:2*N]
    end

    clusterspecs = GroupViaClustering(optimal_radius_method = "silhouettes", max_used_features = 500, use_mmap = true)
    mapper = AttractorsViaFeaturizing(ds, featurizer, clusterspecs; T = 400, Ttr = 600)

    sampler, = statespace_sampler(Random.MersenneTwister(1234);
        min_bounds = [-pi*ones(N) -pi*ones(N)], max_bounds = [pi*ones(N) pi*ones(N)]
    )

    group_cont = GroupAcrossParameterContinuation(mapper)
    Kidx = :K
    Krange = range(0., 10; length = 20)
    fractions_curves, attractors_info = continuation(
                group_cont, Krange, Kidx, sampler;
                show_progress = true, samples_per_parameter = Ns)

    return @strdict(fractions_curves, attractors_info, Krange)
end


Ns = 1000
Nd = 10
params = @strdict Ns Nd
data, file = produce_or_load(
    datadir("data/basins_fractions"), params, continuation_problem;
    prefix = "kur_mcbb", storepatch = false, suffix = "jld2", force = true
)
@unpack fractions_curves,Krange = data

include("figs_continuation_kuramoto.jl")
   
rmap = Attractors.retract_keys_to_consecutive(fractions_curves)
for df in fractions_curves
    swap_dict_keys!(df, rmap)
end

fn = splitext(basename(file) )
plot_filled_curves(fractions_curves, Krange,string(fn[1], ".png")) 

