using DrWatson 
@quickactivate
using Revise
using LightGraphs
using Random
using DifferentialEquations
using ChaosTools
using DynamicalSystems
using JLD2
using Statistics

using CairoMakie


function second_order_kuramoto!(du, u, p, t)
    N = p[1]; α = p[2]; K = p[3]; incidence = p[4]; P = p[5];   
    du[1:N] .= u[1+N:2*N]
    du[N+1:end] .= P .- α .* u[1+N:2*N] .- K .* (incidence * sin.(incidence' * u[1:N]))
end

function get_attractors(K,Nt)

	seed = 5386748129040267798
	Random.seed!(seed)
	# Set up the parameters for the network
	N = 30 # in this case this is the number of oscillators, the system dimension is twice this value
	g = random_regular_graph(N, 3)
	E = incidence_matrix(g, oriented=true)
	P = [isodd(i) ? +1. : -1. for i = 1:N]
	#K = 2.
        ds = ContinuousDynamicalSystem(second_order_kuramoto!, zeros(2*N), [N, 0.1, K, E, vec(P)], (J,z0, p, n) -> nothing)
        diffeq = (alg = Tsit5(), reltol = 1e-9, maxiters = 1e6)

        # get the equilibrium state of the system after a long transient.
        uu = trajectory(ds, 1500; Δt = 0.1, diffeq)
        Δϕ = uu[end][1:N]; Δω = uu[end][N+1:2N]; 
        
        _complete(y) = (length(y) == N) ? [Δϕ; Δω] : y; 
        _proj_state(y) = y[N+1:2*N]
        psys = projected_integrator(ds, _proj_state, _complete; diffeq)
        yg = range(-17, 17; length = 31)
        grid = ntuple(x -> yg, dimension(psys))
	mapper = AttractorsViaRecurrences(psys, grid; sparse = true, Δt = .1,   
            diffeq, 
            mx_chk_fnd_att = 100,
            mx_chk_loc_att = 100,
            safety_counter_max = Int(1e5),
            Ttr = 400.)

	ics = []
	for k = 1:Nt
		u = vec([pi.*rand(N) (rand(N) .- 0.5).*12])
		@show l = mapper(u)
		# push!(ics, ([psys.complete_state; u],l))
		push!(ics, l)
end

	return ics, mapper.bsn_nfo.attractors
end

function continuation_problem()

	seed = 5386748129040267798
	Random.seed!(seed)
	# Set up the parameters for the network
	N = 30 # in this case this is the number of oscillators, the system dimension is twice this value
	g = random_regular_graph(N, 3)
	E = incidence_matrix(g, oriented=true)
	P = [isodd(i) ? +1. : -1. for i = 1:N]
        K = 1.
        ds = ContinuousDynamicalSystem(second_order_kuramoto!, zeros(2*N), [N, 0.1, K, E, vec(P)], (J,z0, p, n) -> nothing)
        diffeq = (alg = Tsit5(), reltol = 1e-9, maxiters = 1e6)

        # get the equilibrium state of the system after a long transient.
        uu = trajectory(ds, 1500; Δt = 0.1, diffeq)
        Δϕ = uu[end][1:N]; Δω = uu[end][N+1:2N]; 
        
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

	return fractions_curves, attractors_info
end



function print_statistics(fractions, figurename)
    
    # ff = Vector{Dict{Int64, Float64}}()
    ff = deepcopy(fractions)
# We rearrange the fractions and we sweep under the carpet the attractors with 
# less the 5% of basin fraction. They are merged under the label -1
    for (n,e) in enumerate(fractions)
        vh = Dict();
        d = sort(e; byvalue = true)
        v = collect(values(d))
        k = collect(keys(d))
        ind = findall(v .> 0.05)
        for i in ind; push!(vh, k[i] => v[i]); end        
        ind = findall(v .<= 0.05)
        if length(ind) > 0 
            push!(vh, -1 => sum(v[ind]))
        end
        # push!(ff, vh)
        ff[n] = vh
    end

    K = range(0., 10; length = 20)
    x = Vector{Float64}()
    h = Vector{Float64}()
    grp = Vector{Int16}()

    for j in 1:length(ff)
        # Position on axis
        push!(x, ones(length(ff[j]))*K[j]...)
        # height values 
        push!(h, values(ff[j])...)
        # group names
        push!(grp, keys(ff[j])...)
    end
    @show x,h,grp
    # rename keys from 0 to N for pretty printing
    for (j,k) in enumerate(unique(grp))
        ind = findall(grp .== k)
        grp[ind] .= j
    end

    fig = Figure()
    ax = Axis(fig[1,1], xlabel = "coupling K", ylabel = "Relative Basin Volume")
    barplot!(x,h, stack = grp, color = grp.+1) 
    save(string(projectdir(), "/plots/a/", figurename),fig)
end


function plot_filled_curves(fractions, figurename)
    ff = deepcopy(fractions)
# We rearrange the fractions and we sweep under the carpet the attractors with 
# less the 5% of basin fraction. They are merged under the label -1
    for (n,e) in enumerate(fractions)
        vh = Dict();
        d = sort(e; byvalue = true)
        v = collect(values(d))
        k = collect(keys(d))
        ind = findall(v .> 0.05)
        for i in ind; push!(vh, k[i] => v[i]); end        
        ind = findall(v .<= 0.05)
        if length(ind) > 0 
            push!(vh, -1 => sum(v[ind]))
        end
        # push!(ff, vh)
        ff[n] = vh
    end
    fractions_curves = ff

    ukeys = ChaosTools.unique_keys(fractions_curves)
    ps = 1:length(fractions_curves)

    bands = [zeros(length(ps)) for k in ukeys]
    for i in eachindex(fractions_curves)
        for (j, k) in enumerate(ukeys)
            bands[j][i] = get(fractions_curves[i], k, 0)
        end
    end
# transform to cumulative sum
    for j in 2:length(bands)
        bands[j] .+= bands[j-1]
    end

    fig = Figure(resolution = (600, 500))
    ax = Axis(fig[1,1])
    for (j, k) in enumerate(ukeys)
        if j == 1
            l, u = 0, bands[j]
        else
            l, u = bands[j-1], bands[j]
        end
        band!(ax, ps, l, u; color = Cycled(j), label = "$k")
    end
    ylims!(ax, 0, 1)
    axislegend(ax; position = :lt)
    # display(fig)

    save(string(projectdir(), "/plots/a/", figurename),fig)
# Makie.save("lorenz84_fracs.png", fig; px_per_unit = 4)
end

# Clustering with Threshold Inf (Euclidean)  and samples taken from uniform dist [-π,π].
d = load(string(projectdir(), "/data/basins/cont_kur_mcbb_samp_pi_pi.jld2"))
f = d["fractions"]
plot_filled_curves(f, "kur_mcbb_pi_pi_threshold_inf.png")


# Clustering with Threshold 1. (Eunclidean norm)  and samples taken from uniform dist [-π,π].
d = load(string(projectdir(), "/data/basins/cont_kur_mcbb_samp_pi_pi_radius_1.jld2"))
f = d["fractions"]
plot_filled_curves(f, "kur_mcbb_pi_pi_threshold_1.png")


# Clustering with Threshold 1. (Hausdorff norm)  and samples taken from uniform dist [-π,π].
d = load(string(projectdir(), "/data/basins/cont_kur_mcbb_samp_pi_pi_radius_1_hausdorff.jld2"))
f = d["fractions"]
plot_filled_curves(f, "kur_mcbb_pi_pi_threshold_1_hausdorff.png")

@load "fraction_test_continuation_kur.jld2"
plot_filled_curves(f, "kur_mcbb_continuation_method.png")
