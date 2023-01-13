using DrWatson
@quickactivate
# using DynamicalSystems
using BenchmarkTools
using Random
using DataFrames
using Attractors
# Define generic timing function that runs benchmarks and saves data
# u0s is Vector{Pair}
function benchmark_mappers_and_save(ds, u0s, grid, featurizer, system;
        ε = nothing, clustering_threshold = 0.0, diffeq = NamedTuple(),
        Ttr = 100, Ns = 2 .^ (6:2:12), force = (system, method) -> false,
        recurrence_kwargs = (;), clustering_config = GroupViaClustering(),
    )

    sampler, = statespace_sampler(Random.MersenneTwister(1234);
        min_bounds = minimum.(grid), max_bounds = maximum.(grid)
    )

    ics_container = [Dataset([sampler() for i in 1:N]) for N in Ns]

    # reusable comparison function which is given to `produce_or_load`
    function mapper_benchmark(config)
        # `system, D` aren't used, but we want them in output
        (; mapper, method, N, system, D) = config
        # We use `i` because we want all mappers to use same initial conditions!
        @info "N = $N"
        i = findfirst(isequal(N), Ns)
        b = @benchmark(
            basins_fractions($(mapper), $(ics_container[i]); show_progress=false),
            seconds = 10, # max benchmarking time (will take more if run costs more time)
        )
        b = median(b)
        time = b.time/1e9
        allocs = b.allocs
        outdata = @strdict(system, method, N, time, allocs, D)
        @info "Median time (sec) = $("time")"
        return outdata
    end

    mappers = []
    # Alrighty, now we go through all attractor mapper methods

    known_attractors = Dict(
        k => trajectory(ds, 10000, v; Δt = 1, Ttr) for (k,v) in u0s if k ≠ -1
    )
    mapper = AttractorsViaProximity(ds, known_attractors, ε; diffeq, Ttr)
    push!(mappers, (mapper, "proximity"))

    mapper = AttractorsViaRecurrences(ds, grid;
    diffeq, show_progress = false, Ttr, sparse = false, recurrence_kwargs...)
    push!(mappers, (mapper, "recurrences"))

    mapper = AttractorsViaRecurrences(ds, grid;
    diffeq, show_progress = false, Ttr, sparse = true, recurrence_kwargs...)
    push!(mappers, (mapper, "recurrences_sparse"))

    # Featurizing, supervised
    # First generate the templates
    # function features_from_u(u)
    #     A = trajectory(ds, 100, u; Ttr, Δt = 1, diffeq)
    #     featurizer(A, 0)
    # end
    # t = [features_from_u(x[2]) for x in u0s]
    # templates = Dict([u0[1] for u0 ∈ u0s] .=> t) #keeps labels of u0s
    # clusterspecs = GroupViaClustering(; templates, clustering_threshold)
    # mapper = AttractorsViaFeaturizing(ds, featurizer, clusterspecs; Ttr, diffeq)
    # push!(mappers, (mapper, "featurizing_supervised"))

    # Featurizing, unsupervised
    mapper = AttractorsViaFeaturizing(ds, featurizer, clustering_config;
    diffeq, Ttr, threaded = false)
    push!(mappers, (mapper, "featurizing_cluster_single"))

    mapper = AttractorsViaFeaturizing(ds, featurizer, clustering_config;
    diffeq, Ttr, threaded = true)
    push!(mappers, (mapper, "featurizing_cluster_thread"))

    clustering_config_og = GroupViaClustering(; num_attempts_radius = 200,
        silhouette_statistic = minimum, optimal_radius_method = "silhouettes")
    mapper = AttractorsViaFeaturizing(ds, featurizer, clustering_config_og;
    diffeq, Ttr, threaded = false)
    push!(mappers, (mapper, "featurizing_cluster_original"))
    for (mapper, method) in mappers
        @info "Benchmarking $(method)..."
        for N in Ns
            D = dimension(ds)
            config = (; mapper, method, N, system, D)
            @produce_or_load(
                mapper_benchmark, config, datadir("comparison_benchmarks", "mappers");
                force = force(system, method), verbose = false,
            )
        end
    end

    return
end

# %%
# # Run benchmarks
# Henon map:
let
    ds = Systems.henon(zeros(2); a = 1.4, b = 0.3)
    u0s = [1 => [0.0, 0.0], -1 => [0.0, 2.0]]
    xg = yg = range(-2.0, 2.0; length=100)
    grid = (xg, yg)
    function featurizer(A, t)
        x = [mean(A[:, 1]), mean(A[:, 2])]
        return any(isinf, x) ? [200.0, 200.0] : x
    end

    benchmark_mappers_and_save(
        ds, u0s, grid, featurizer, "henon";
        ε = 0.01, clustering_threshold = 20,
        Ns = 2 .^ (4:2:16)
    )
end

# %%
# Lorenz84
let
    using OrdinaryDiffEq
    F = 6.886; G = 1.347; a = 0.255; b = 4.0
    ds = Systems.lorenz84(; F, G, a, b)
    u0s = [
        1 => [2.0, 1, 0], # periodic
        2 => [-2.0, 1, 0], # chaotic
        3 => [0, 1.5, 1.0], # fixed point
    ]
    diffeq = (alg = Vern9(), reltol = 1e-9, abstol = 1e-9)
    M = 200; z = 3
    xg = yg = zg = range(-z, z; length = M)
    grid = (xg, yg, zg)

    function featurizer(A, t)
        # This is the number of boxes needed to cover the set
        probs = probabilities(ValueHistogram(RectangularBinning(0.1)), A)
        g = exp(entropy(Renyi(0), probs))
        return [g, minimum(A[:,1])]
    end

    recurrence_kwargs = (; mx_chk_att = 20)

    benchmark_mappers_and_save(
        ds, u0s, grid, featurizer, "lorenz84";
        ε = 0.01, clustering_threshold = 20,
        Ns = 2 .^ (4:2:14), recurrence_kwargs, diffeq,
        force = (system, method) -> system != "lorenz84",
    )
end


# %%
# Kuramoto:
let
    error()
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
    K = 1.0

    ds = ContinuousDynamicalSystem(second_order_kuramoto!, zeros(2*D), [D, 0.1, K, E, vec(P)], (J, z0, p, n) -> nothing)
    diffeq = (alg = Tsit5(), reltol = 1e-9, maxiters = 1e6)

    # TODO: What featurizer to use here? What do MCBB.jl uses?
    function featurizer(A, t)
        return [mean(A[:, i]) for i in D+1:2*D]
    end
    
    ## Continuation clustering (MCBB)
    #
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

    ## Continuation recurrences
    ##
    _complete(y) = (length(y) == N) ? [Δϕ; Δω] : y; 
    _proj_state(y) = y[N+1:2*N]
    psys = projected_integrator(ds, _proj_state, _complete; diffeq)
    yg = range(-17, 17; length = 31)
    grid = ntuple(x -> yg, dimension(psys))

    mapper = AttractorsViaRecurrences(psys, grid; sparse = true, Δt = .1,   
        diffeq, 
        mx_chk_fnd_att = 100,
        mx_chk_loc_att = 100,
        Ttr = 400.)

    sampler, = statespace_sampler(Random.MersenneTwister(1234);
        min_bounds = [-pi*ones(N) -12*ones(N)], max_bounds = [pi*ones(N) 12*ones(N)]
        )
    continuation = RecurrencesSeedingContinuation(mapper)
    Kidx = 3
    Krange = range(0., 10; length = 20)
    fractions_curves, attractors_info = basins_fractions_continuation(
        continuation, Krange, Kidx, sampler;
        show_progress = true, samples_per_parameter = 10000
    )

    # TODO: I also need to map initial conditions to attractors for benchmarking the
    # Proximity and Supervised Clustering methods

    benchmark_suite["henon"]  = benchmark_mappers(
        ds, u0s, grid, featurizer; ε=0.01, clustering_threshold = 20,
        Ns = 2 .^ (6:2:8)
    )
end
