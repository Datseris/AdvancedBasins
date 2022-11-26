using DrWatson
@quickactivate
using OrdinaryDiffEq:Vern9
using DynamicalSystemsBase
using Attractors, Random
using CairoMakie
include("$(srcdir())/vis/basins_plotting.jl")

monod(r, R, K) = r*R/(K+R)
function μ!(μs, rs, Rs, Ks)
    for i in eachindex(μs)
        mo1 = monod(rs[i], Rs[1], Ks[1,i])
        mo2 = monod(rs[i], Rs[2], Ks[2,i])
        mo3 = monod(rs[i], Rs[3], Ks[3,i])
        μs[i] = min(mo1, mo2, mo3)
    end
    nothing
end
#not the most optimied but w/e
function Rcoup!(Rcoups, Ns, Rs, μs, cs)
    fill!(Rcoups, 0.0)
    for j in eachindex(Rcoups)
        for i in eachindex(μs)
            Rcoups[j] += cs[j,i] * μs[i] * Ns[i]
        end
    end
    nothing
end

function competition!(du, u, p, t)
    @unpack rs, Ks, ms, Ss, cs, μs, Rcoups, D = p
    n = size(Ks, 2)
    Ns = view(u, 1:n)
    Rs = view(u, n+1:n+3)
    dNs = view(du, 1:n)
    dRs = view(du, n+1:n+3)
    μ!(μs, rs, Rs, Ks)
    Rcoup!(Rcoups, Ns, Rs, μs, cs)
    @. dNs = Ns * (μs - ms)
    @. dRs = D*(Ss - Rs) - Rcoups
    nothing
end

mutable struct CompetitionDynamics3
    rs :: Vector{Float64}
    ms :: Vector{Float64}
    Ss :: Vector{Float64}
    μs :: Vector{Float64}
    Rcoups :: Vector{Float64}
    Ks :: Matrix{Float64}
    cs :: Matrix{Float64}
    D :: Float64
end

function CompetitionDynamics(fig="1")
    if fig == "4" || fig == "1"
        Ks  = [
            0.20 0.05 0.50 0.05 0.50 0.03 0.51 0.51;
            0.15 0.06 0.05 0.50 0.30 0.18 0.04 0.31;
            0.15 0.50 0.30 0.06 0.05 0.18 0.31 0.04;
        ]

        cs = [
            0.20 0.10 0.10 0.10 0.10 0.22 0.10 0.10;
            0.10 0.20 0.10 0.10 0.20 0.10 0.22 0.10;
            0.10 0.10 0.20 0.20 0.10 0.10 0.10 0.22;
        ]
        if fig == "1"
            Ks = Ks[:, 1:5]
            cs = cs[:, 1:5]
        end
    elseif fig == "2" || fig == "3"
        Ks = [
            0.20 0.05 1.00 0.05 1.20;
            0.25 0.10 0.05 1.00 0.40;
            0.15 0.95 0.35 0.10 0.05;
        ]

        cs = [
            0.20 0.10 0.10 0.10 0.10;
            0.10 0.20 0.10 0.10 0.20;
            0.10 0.10 0.20 0.20 0.10;
        ]

    else
        @error "nope"
    end

    N = size(Ks, 2)

    rs = [1.0 for i=1:N]
    D = 0.25
    ms = [D for i=1:N]
    Ss = [10.0 for j=1:3]
    μs = zeros(Float64, N)
    Rcoups = zeros(Float64, 3)
    return CompetitionDynamics3(rs, ms, Ss, μs, Rcoups, Ks, cs, D)
end

function plot_attractors(atts; fig=nothing, ax=nothing, idxs=[1,2,3])
    if isnothing(fig) fig = Figure(res=(1000, 1000)) end
    if isnothing(ax) ax = Axis3(fig[1,1]) end
    for (k,att) in atts
        if length(att) == 1
            scatter!(ax, att[:,idxs[1]], att[:,idxs[2]], att[:, idxs[3]], markersize=20)
        else
            lines!(ax, att[:, idxs[1]], att[:, idxs[2]], att[: ,idxs[3]])
        end
    end
    return fig, ax
end


function plot_dynamics(p)
    fig = Figure(resolution=(800, 800))
    ax = Axis(fig[1,1])
    # xlims!(1000, 2000)
    for i=1:N lines!(ax, t, tr[:,i], label="$i") end
    axislegend("Unit", position = :rt, orientation = :horizontal)
    ax2 = Axis3(fig[2,1])
    lines!(ax2, tr[:,1], tr[:,2], tr[:,3])
    fig
    return fig, t, tr, p
end

reduced_grid(grid, newlength) = map(g -> range(minimum(g), maximum(g); length = newlength), grid)


figidx = "2"
p = CompetitionDynamics(figidx)

N = size(p.Ks, 2)
u0 = [[0.1 for i=1:N]; [S for S in p.Ss]]
ds = ContinuousDynamicalSystem(competition!, u0, p, (J, z, p, t)->nothing)
int = integrator(ds, u0)

# -------------------------- Step 1: replicate paper ------------------------- #
T = 2000.0; Ttr = 0.0; Δt = 0.5;
tr = trajectory(int, T; Ttr, Δt); t=Ttr:Δt:T;
fig, ax = plot_dynamics(p)
fig
save("$(plotsdir())/populationdynamics-fig$figidx.png", fig, px_per_unit=3)


# ------------- Step 2: Recurrences for a single parameter ------------- #
diffeq = (alg = Vern9(), maxiters=Inf);
xg = range(0, 60,length = 300);
grid = ntuple(x->xg, N+3);
ds = ContinuousDynamicalSystem(competition!, u0, p, (J, z, p, t)->nothing)
mapper = AttractorsViaRecurrences(ds, grid;
        # mx_chk_fnd_att = 1000,
        # mx_chk_loc_att = 500,
        # mx_chk_att = 100,
        Δt= 1.0,
        diffeq,
        stop_at_Δt=true,
    );

#option 1
redugrid = reduced_grid(grid, 2);
basins, atts = basins_of_attraction(mapper, redugrid);

#option 2
basins = zeros(Int32, map(length, redugrid))
I = CartesianIndices(basins)
ics = Dataset([Attractors.generate_ic_on_grid(redugrid, i) for i in vec(I)])
fs, labels, atts = basins_fractions(mapper, ics)

fig, ax = plot_attractors(atts; idxs=[1,2,3])
fig

# have_converged = verify_convergence_attractors(ds, atts, 50)

# ------------------------ Step 3: apply continuation ------------------------ #
# using DynamicalSystems,
using ProgressMeter
pidx = :D; ps = 0.2:0.01:0.3;
# pidx = :D; ps = 0.2:0.01:0.3;
unitidx = 4
isextinct(A, idx) = all(A[:, idx] .<= 1e-2)
distance_extinction = function(A,B, idx)
    A_extinct = isextinct(A,idx)
    B_extinct = isextinct(B,idx)
    return Int32(A_extinct && B_extinct)
end

ds = ContinuousDynamicalSystem(competition!, u0, p, (J, z, p, t)->nothing)
mapper = AttractorsViaRecurrences(ds, grid;
        Δt= 1.0,
        diffeq,
        stop_at_Δt=true,
    );

#default
metric = Euclidean(); threshold = Inf; info_extraction(A) = A; methodname = "default"

#custom
metric = (A, B) -> distance_extinction(A, B, unitidx);
threshold = Inf; methodname = "extinction"
info_extraction(A) = isextinct(A, unitidx)

continuation = RecurrencesSeedingContinuation(mapper; metric, threshold, info_extraction);
sampler, = statespace_sampler(Random.MersenneTwister(1234);
    min_bounds = minimum.(grid), max_bounds = maximum.(grid)
);
fractions_curves, attractors_info = basins_fractions_continuation(
    continuation, ps, pidx;
    show_progress = true, samples_per_parameter = 100
)

fig, ax = basins_fractions_plot(fractions_curves, collect(ps), add_legend=true)
ax.xlabel = "D"
ax.ylabel = "fractions"
vlines!(ax, 0.25, color=:red)
fig
save("$(plotsdir())/populationdynamics-fractionscontinuation-fig$figidx-metric_$(metric)-threshold-$(threshold).png", fig, px_per_unit=3)

#find which attractor label corresponds to extinct X
attractors_info
fractions_curves
attractors_info[1][4][:,4]
isextinct(attractors_info[1][1], 4)


# ----------------------------- Step 4: Grouping ----------------------------- #
"""
Group attractors based on features from featurizer and then change their labels so that
the grouped attractors have the same label.
"""
function match_by_grouping(groupingconfig, atts, featurizer; fs=nothing)
    features = [featurizer(A) for (k, A) in atts]
    labels = group_features(features, groupingconfig)
    rmap = Dict(keys(atts) .=> labels)
    swap_dict_keys!(atts, rmap)
    return rmap
end

"""
Given a replacement map that is a dictionary mapping the old keys in `fs` to new keys,
return the updated fractions `fs` that correspond to the sum of the values of `fs` with
the same key.
"""
function sum_fractions_keys(fs, rmap)
    newkeys = unique(values(rmap))
    fssum = Dict(newkeys .=> 0.0)
    for oldkey in keys(rmap)
        newkey = rmap[oldkey]
        fssum[newkey] += fs[oldkey]
    end
    return fssum
end

groupingconfig = GroupViaClustering(; min_neighbors=1, optimal_radius_method=0.5) #note that minneighbors = 1 is crucial for grouping single attractors
featurizer(A) = [Int32(isextinct(A, unitidx))]
atts = deepcopy(attractors_info[6])
fs = deepcopy(fractions_curves[6])

features_outside = [featurizer(A) for (k, A) in atts]
rmap = match_by_grouping(groupingconfig, atts, featurizer; fs)
atts
rmap
fs = sum_fractions_keys(fs, rmap)