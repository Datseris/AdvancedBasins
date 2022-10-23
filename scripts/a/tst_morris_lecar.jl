using DrWatson
@quickactivate 
using OrdinaryDiffEq:Vern9
using Random
using DynamicalSystems
# using CairoMakie
using LaTeXStrings

function morris_lecar!(du, u, p, t)
    V1 = -0.00; V2 = 0.15; V4 = 0.1; VCa = 1 ;  VL = -0.5; VK = -0.7; gCa = 1.2; gK = 2; gL = 0.5; τ = 3; 
    I, V3 = p 
    V, N = u
    M(x) = 0.5*(1 + tanh((x-V1)/V2))
    G(x) = 0.5*(1 + tanh((x-V3)/V4))
    du[1] = -gCa*M(V)*(V - VCa) -gK*N*(V - VK) -gL*(V-VL) + I 
    du[2] = 1/τ*(-N + G(V))
end


function _get_basins_morris_lecar(d)
    @unpack V3, I, res = d
    # I = 0.5; V3 = 0.12
    p = [I, V3]
    xg = yg = range(-1,1,length = 1000)
    df = ContinuousDynamicalSystem(morris_lecar!,rand(2), p)
    diffeq = (reltol = 1e-9,  alg = Vern9())
    mapper = AttractorsViaRecurrences(df, (xg, yg);
            mx_chk_fnd_att = 1000, 
            mx_chk_loc_att = 1000, 
            # mx_chk_att = 2,
             sparse = true, Ttr = 10)
    xg = yg = range(-1, 1,length=res)
    basins, att = basins_of_attraction(mapper, (xg, yg))
    return @strdict(basins, xg, yg)
end


function continuation_test(V3)
    I = 0.5e-3; 
    p = [I, V3]
    xg = yg = range(-1,1,length = 10000)
    df = ContinuousDynamicalSystem(morris_lecar!,rand(2), p)
    diffeq = (reltol = 1e-9,  alg = Vern9(), maxiters = 1e9)
    mapper = AttractorsViaRecurrences(df, (xg, yg);
            mx_chk_fnd_att = 400, 
            mx_chk_loc_att = 400, 
            sparse = true, Δt = 0.01, diffeq, safety_counter_max = Int(1e6), 
            show_progress = true)
    sampler, = ChaosTools.statespace_sampler(Random.MersenneTwister(1234);
        min_bounds = [-0.5, -1], max_bounds = [0.5, 1]
    )
    Irange = range(0., 0.35; length = 50)
    Iidx = 1
    continuation = RecurrencesSeedingContinuation(mapper; threshold = 1., metric = Euclidean())
    fractions_curves, attractors_info = basins_fractions_continuation(
        continuation, Irange, Iidx, sampler;
        show_progress = true, samples_per_parameter = 100
    )

    bf = print_bif(Irange, fractions_curves, attractors_info)

    return fractions_curves, attractors_info, bf
end


function print_bif(K, frac, att)

    bf = []
    for (k,a) in enumerate(att)
        for b in a
            # @show b[1]
            v = [b[1]*ones(length(b[2])), K[k]*ones(length(b[2])), Matrix(b[2])[:,1]]
            push!(bf, hcat(v...))
        end
    end
    return hcat(bf'...)
end

# bf = print_bif(range(0, .1, 20) ,f, a)

function print_fig(w,h, V3, I, res)

    # res = 1500
    data, file = produce_or_load(
        datadir("basins"), # path
        @dict(res, V3, I), # container
        _get_basins_morris_lecar, # function
        prefix = "basin_morris_lecar", # prefix for savename
        force = true
    )
    @unpack basins, xg, yg = data

    fig = Figure(resolution = (w, h))
    ax = Axis(fig[1,1], ylabel = L"y_0", xlabel = L"x_0", yticklabelsize = 30, 
            xticklabelsize = 30, 
            ylabelsize = 30, 
            xlabelsize = 30, 
            xticklabelfont = "cmr10", 
            yticklabelfont = "cmr10")
    # heatmap!(ax, xg, yg, basins, rasterize = 1, colormap = cmap)
    heatmap!(ax, xg, yg, basins, rasterize = 1)
    save("../plots/basins_morris_lecar.png",fig)
end

# Fraction continuation and Bifurcation Diagram
V3 = 0.1; I = 0.08
f,a,b = continuation_test(V3)
# print_fig(600,500, V3, I, 100); 
using Plots
# Plot attractors with different colors. 
p = scatter([0,0],[0,0])
for att in unique(b[1,:])
    ind = findall(att .== b[1,:])
    scatter!(p, b[2,ind], b[3,ind])
end
p
