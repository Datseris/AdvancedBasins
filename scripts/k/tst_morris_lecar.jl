using DrWatson
@quickactivate
using OrdinaryDiffEq:Vern9
using OrdinaryDiffEq:AutoTsit5
using OrdinaryDiffEq:Rosenbrock23
using OrdinaryDiffEq:Tsit5
using Random
# using Attractors
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
    xg = yg = range(-1,1,length = 10000)
    # xg = yg = range(-1, 1,length=res)
    basins, att = basins_of_attraction(mapper, (xg, yg))
    return @strdict(basins, xg, yg)
end


function continuation_test(V3)
    I = 0.5e-3;
    p = [I, V3]
    xg = yg = range(-1,1,length = 20000)
    df = ContinuousDynamicalSystem(morris_lecar!,rand(2), p)
    diffeq = (reltol = 1e-9,  alg = Vern9(), maxiters = 1e9)
    # diffeq = (reltol = 1e-9,  alg = AutoTsit5(Rosenbrock23()), maxiters = 1e9)
    # diffeq = (reltol = 1e-9,  alg = Tsit5(), adaptative=false, dt=0.0005, maxiters = 1e9)
    mapper = AttractorsViaRecurrences(df, (xg, yg);
            mx_chk_fnd_att = 50000,
            mx_chk_loc_att = 1000,
            # mx_chk_att = 100,
            sparse = true, Δt = 0.01, diffeq, safety_counter_max = Int(1e7),
            show_progress = true, Ttr=100)
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

    return fractions_curves, attractors_info, bf, Irange
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
    Makie.heatmap!(ax, xg, yg, basins, rasterize = 1)
    Makie.save("$(plotsdir())/basins_morris_lecar.png",fig)
end


using CairoMakie
# Fraction continuation and Bifurcation Diagram
V3 = 0.1; I = 0.08
f,a,b, Irange = continuation_test(V3)
# Plot attractors with different colors.
fig = Figure()
ax = Axis(fig[1,1])
for att in unique(b[1,:])
    ind = findall(att .== b[1,:])
    Makie.scatter!(ax, b[2,ind], b[3,ind])
end
fig

# print_fig(w, h, V3, I, res)
print_fig(600,500, V3, I, 100);

colors = [:blue, :yellow, :orange, :green, :magenta, :brown, :cyan]
unique_keys = unique(keys.(a))
# for unique_key in unique_keys
    # idx_a = findlast(x->x==unique_key, keys.(a))
for idx_a = 1:length(a)
    atts = a[idx_a]
    fig = Figure(resolution=(500,500))
    ax = Axis(fig[1,1], title="I = $(Irange[idx_a]), idx_a = $idx_a")
    for (k, att) in atts
    Makie.scatter!(ax, att[:,1], att[:,2], color=colors[k])
    end
    Makie.xlims!(ax, -0.6, 0.4)
    Makie.ylims!(ax, -0.2, 0.6)
    # Makie.save("$(plotsdir())/morrislecar/attractors_morris_lecar_idxa_$(idx_a).png",fig)
    Makie.save("plots/morrislecar/attractors_morris_lecar_idxa_$(idx_a).png",fig)
end


#FP and LC do seem stable
# ic_fp = a[26][3][1] .+ 1e-3
idx = 15
ic = a[idx][1][1]
I = Irange[idx];
p = [I, V3]
df = ContinuousDynamicalSystem(morris_lecar!,rand(2), p)
diffeq = (reltol = 1e-9,  solver = Tsit5(), adaptative=false, dt=0.0005, maxiters = 1e9)
T = 1000;
tr = trajectory(df, T, ic; Ttr=0, diffeq, Δt=0.01)
ts = 0:0.01:T
fig = Figure()
ax = Axis(fig[1,1])
Makie.scatter!(ax, tr[:,1], tr[:,2])
ax = Axis(fig[2,1])
lines!(ax, ts[end-500:end], tr[end-500:end,1])
# lines!(ax, a[idx][1][:,1], a[13][1][:,2])
fig


p = [I, V3]
xg = yg = range(-1,1,length = 10000)
df = ContinuousDynamicalSystem(morris_lecar!,rand(2), p)
diffeq = (reltol = 1e-9,  alg = Vern9())
mapper = AttractorsViaRecurrences(df, (xg, yg);
        mx_chk_fnd_att = 50000,
        mx_chk_loc_att = 50000,
        # mx_chk_att = 2,
         sparse = true, Ttr = 10, Δt=0.1)
xg = yg = range(-1, 1,length=10)
basins, att = basins_of_attraction(mapper, (xg, yg))
