using DrWatson 
@quickactivate
using Revise
using Attractors
using Random
using Graphs
# using OrdinaryDiffEq:Tsit5
using Statistics:mean
using PredefinedDynamicalSystems


function continuation_problem(di)
    @unpack Nd, Ns = di
    K = 3.; ω = range(-1, 1; length = Nd)
    # diffeq = (alg = Tsit5(), reltol = 1e-9, maxiters = 1e6)
    ds = Systems.kuramoto(Nd; K = K, ω = ω)

    function featurizer(A, t)
        u = A[end,:]
        return abs(mean(exp.(im .* u)))
    end

    clusterspecs = GroupViaHistogram(FixedRectangularBinning(range(0., 1.; step = 0.2), 1))
    mapper = AttractorsViaFeaturizing(ds, featurizer, clusterspecs; T = 200)

    sampler, = statespace_sampler(Random.MersenneTwister(1234);
        min_bounds = -pi*ones(Nd), max_bounds = pi*ones(Nd)
    )

    group_cont = GroupAcrossParameterContinuation(mapper)
    Kidx = :K
    # Krange = 10 .^range(-2,log10(2), length = 20) 
    Krange = range(0., 2; length = 40)
    fractions_curves, attractors_info = continuation(
                group_cont, Krange, Kidx, sampler;
                show_progress = true, samples_per_parameter = Ns)

    return @strdict(fractions_curves, attractors_info, Krange)
end

Ns = 2000
Nd = 10
params = @strdict Ns Nd
data, file = produce_or_load(
    datadir("data/basins_fractions"), params, continuation_problem;
    prefix = "kur_hist", storepatch = false, suffix = "jld2", force = false
)
@unpack fractions_curves,Krange = data

rmap = Attractors.retract_keys_to_consecutive(fractions_curves)
for df in fractions_curves
    swap_dict_keys!(df, rmap)
end

include("figs_continuation_kuramoto.jl")

fn = splitext(basename(file))
plot_filled_curves_kuramoto(fractions_curves, Krange,string(fn[1], ".png")) 


