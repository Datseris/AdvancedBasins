using DrWatson 
@quickactivate
using Revise
# using DynamicalSystems
using Attractors
using Random
using Graphs
using OrdinaryDiffEq
using Statistics:mean
using JLD2


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
    @unpack Nd, Ns, thr, res = di

	# Set up the parameters for the network
	N = Nd # in this case this is the number of oscillators, the system dimension is twice this value
        p = KuramotoParameters(; N)
        diffeq = (alg = Vern9(), reltol = 1e-9, maxiters = 1e8)
        ds = CoupledODEs(second_order_kuramoto!, zeros(2*N), p; diffeq)

        _complete(y) = (length(y) == N) ? zeros(2*N) : y; 
        _proj_state(y) = y[N+1:2*N]
        psys = ProjectedDynamicalSystem(ds, _proj_state, _complete)
        yg = range(-12, 12; length = res)
        grid = ntuple(x -> yg, dimension(psys))
	mapper = AttractorsViaRecurrences(psys, grid; sparse = true, Δt = 0.01,   
            show_progress = true, mx_chk_fnd_att = 100,
            mx_chk_safety = Int(1e7),
            force_non_adaptive = true,
            mx_chk_loc_att = 10)#,
            #Ttr = 400.)

        sampler, = statespace_sampler(Random.MersenneTwister(1234);
            min_bounds = [-pi*ones(N) -pi*ones(N)], max_bounds = [pi*ones(N) pi*ones(N)]
        )

        cont_rec = RecurrencesSeededContinuation(mapper; threshold = thr)
        Kidx = :K
        Krange = range(0., 10.; length = 40)
        fractions_curves, attractors_info = continuation(
            cont_rec, Krange, Kidx, sampler;
            show_progress = true, samples_per_parameter = Ns
        )
	return @strdict(fractions_curves, attractors_info, Krange)
end


Ns = 1000
Nd = 10
res = 51
thr = Inf
params = @strdict Ns res thr Nd
data, file = produce_or_load(
    datadir("data/basins_fractions"), params, continuation_problem;
    prefix = "kur_non_adap_rec", storepatch = false, suffix = "jld2", force = true
)
@unpack fractions_curves,Krange = data

include("figs_continuation_kuramoto.jl")

fn = splitext(basename(file))
plot_filled_curves(fractions_curves, Krange,string(fn[1], ".png")) 
