using DrWatson 
@quickactivate
using Revise
using Attractors
using Attractors.DynamicalSystemsBase
using Random
using Graphs
using OrdinaryDiffEq:Tsit5
using Statistics:mean
using JLD2


function second_order_kuramoto!(du, u, p, t)
    (; N, α, K, incidence, P) = p
    # N = p[1]; α = p[2]; K = p[3]; incidence = p[4]; P = p[5];
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


function continuation_problem(;thr = Inf)
	# Set up the parameters for the network
	N = 30 # in this case this is the number of oscillators, the system dimension is twice this value
        p = KuramotoParameters(; N)
        ds = ContinuousDynamicalSystem(second_order_kuramoto!, zeros(2*N), p, (J, z0, p, n) -> nothing)
        diffeq = (alg = Tsit5(), reltol = 1e-9, maxiters = 1e6)
        
        _complete(y) = (length(y) == N) ? zeros(2*N) : y; 
        _proj_state(y) = y[N+1:2*N]
        psys = projected_integrator(ds, _proj_state, _complete; diffeq)
        yg = range(-17, 17; length = 61)
        grid = ntuple(x -> yg, dimension(psys))
	mapper = Attractors.AttractorsViaRecurrences(psys, grid; sparse = true, Δt = .1,   
            diffeq, 
            show_progress = true, mx_chk_fnd_att = 100,
            mx_chk_safety = Int(1e5),
            mx_chk_loc_att = 100,
            Ttr = 200.)

        sampler, = statespace_sampler(Random.MersenneTwister(1234);
            min_bounds = [-pi*ones(N) -pi*ones(N)], max_bounds = [pi*ones(N) pi*ones(N)]
        )

        dts_dis(x,y)= dataset_distance(x,y, Centroid())
        gc = GroupViaClustering(;clust_distance_metric = dts_dis, 
                optimal_radius_method = 1., min_neighbors = 1)
        continuation = ClusteringAttractorsContinuation(mapper; par_weight = 1.,
                group_config = gc, cluster_in_slice = 0.)
        Kidx = :K
        Krange = range(0., 10; length = 20)
        fs, att, fsj = basins_fractions_continuation(
                continuation,  Krange, Kidx, sampler; 
                show_progress = true, samples_per_parameter = 4000)

	return fsj, att, Krange
end

f, a, K = continuation_problem(thr = .5)

save("fractions_clust_att_kur.jld2", "f", f, "K", K)
