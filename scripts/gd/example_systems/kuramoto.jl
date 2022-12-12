using DrWatson
@quickactivate
using Attractors, OrdinaryDiffEq, GLMakie, Random
using SparseArrays: sparse
using Graphs: random_regular_graph, incidence_matrix
include(srcdir("vis", "basins_plotting.jl"))
include(srcdir("additional_predefined_systems.jl"))

# for K < 1 you should find one or two attractors (unsynch).
# for 4 < K < 7 : zillions of attractors
# K > 9 one attractor (synchronized).
N = 10
K = 6.0

ds = kuramoto_network_2ndorder(; N, K)
diffeq = (alg = Tsit5(), reltol = 1e-6, abstol = 1e-6, maxiters = Inf)

uu = trajectory(ds, 1500; Δt = 0.1, diffeq)

recurrence_kwargs = (;
    mx_chk_fnd_att = 1000,
    mx_chk_loc_att = 2000,
    Ttr = 200.0,
    mx_chk_safety = Int(1e6),
    diffeq,
)

# %% Mapper that projects into frequencies ω
projection = N+1:2N
complete = y -> vcat(π .* rand(N), y)
yg = range(-17, 17; length = 101)
grid = ntuple(x -> yg, N)

psys = projected_integrator(ds, projection, complete; diffeq)

mapper = AttractorsViaRecurrences(psys, grid; Δt = 0.1, recurrence_kwargs...)

n = 1000
labels = []
ics = []
for k = 1:n
    u = 12(rand(N) .- 0.5)
    l = mapper(u)
    # push!(ics, ([psys.complete_state; u],l))
    push!(labels, l)
    push!(ics, u)
end

att = mapper.bsn_nfo.attractors

fig = Figure()
ax = Axis(fig[1,1])
for (k, a) in att
    scatterlines!(ax, a[:, 1], a[:, 2])
end
display(fig)

ids = sort!(collect(keys(att)))

@show ids

# %% Lyapunov exponents and Order Parameter
function order_parameter(φs)
    return abs(sum(φ -> exp(im*φ), φs))/length(φs)
end

using ChaosTools: lyapunov
using Statistics

Rs = Dict()
for i in 1:n
    l = labels[i]
    haskey(Rs, l) && continue
    @show l
    u = ics[i]
    fullu = vcat(π .* rand(N), u)
    tr = trajectory(ds, 10.0, fullu; Ttr = 100)
    ωs = tr[end, projection]
    # @show ωs
    @show std(ωs)
    # R = order_parameter(tr[end, 1:N])
    phases = tr[:, 1:N]
    R = mean(map(order_parameter, phases))
    @show R
    Rs[l] = R
    λ = lyapunov(ds, 10000.0; u0 = fullu, Ttr = 100.0)
    @show λ
end


# %% continuation
# If we have the recurrences continuation, we can always map it to
# the featurized continuation, as we have the attractors.
projection = N+1:2N
complete = y -> vcat(π .* rand(N), y)
yg = range(-17, 17; length = 101)
grid = ntuple(x -> yg, N)

psys = projected_integrator(ds, projection, complete; diffeq)
prange = range(0, 10; length = 21)
pidx = :K

mapper = AttractorsViaRecurrences(psys, grid; Δt = 0.1, recurrence_kwargs...)

continuation = RecurrencesSeedingContinuation(mapper; threshold = Inf)

fractions_curves, attractors_info = basins_fractions_continuation(
    continuation, prange, pidx;
    show_progress = true, samples_per_parameter = 20
)

fig = basins_fractions_plot(fractions_curves, prange)
display(fig)
GLMakie.save(desktop("original_kuramoto_recurrences.png"), fig)

# %% Aggregate attractors by clustring
using Statistics
# Notice that featurizers for this pipeline don't get `t` because recurrences don't save `t`
function featurizer_kuramoto(A)
    ωs = A[end]
    x = std(ωs)
    y = mean(A[:, 1])
    return SVector(x, y)
end
# function featurizer_kuramoto(A)
#     return [mean(x) for x in columns(A)]
# end
# function featurizer_kuramoto(A)
#     j=1 #special node

#     return [mean(x) for x in columns(A)]
# end

# Okay, here we define the aggregation function that
# takes in the output of `basins_fractions_ontinuation`
# and aggregates it to different dicts
# function aggregate_attractors_and_fractions(...)
# inputs
featurizer = featurizer_kuramoto

# All the following will become one nice function
# "group_attractors" or something like that.

# Set up containers
P = length(prange)
example_feature = featurizer(first(values(attractors_info[1])))
features = typeof(example_feature)[]
original_labels = keytype(first(fractions_curves))[]
parameter_idxs = Int[]
unlabeled_fractions = zeros(P)
# Transform original data into sequential vectors
spp = length(prange)

for i in eachindex(fractions_curves)
    fs = fractions_curves[i]
    ai = attractors_info[i]
    A = length(ai)
    append!(parameter_idxs, (i for _ in 1:A))
    unlabeled_fractions[i] = get(fs, -1, 0.0)
    for k in keys(ai)
        push!(original_labels, k)
        push!(features, featurizer(ai[k]))
    end
end

clust_config = GroupViaClustering(min_neighbors = 10)
par_weight = 0
# TODO: This becomes "group_features"
clustered_labels = Attractors.cluster_all_features(features, clust_config, par_weight)
# okay this finally worked but its results are still rather sad; I get three attractors

# Anyways, time to reconstruct the joint fractions
joint_fractions = [Dict{Int,Float64}() for _ in 1:P]
current_p_idx = 0
for j in eachindex(clustered_labels)
    new_label = clustered_labels[j]
    p_idx = parameter_idxs[j]
    if p_idx > current_p_idx
        current_p_idx += 1
        joint_fractions[current_p_idx][-1] = unlabeled_fractions[current_p_idx]
    end
    d = joint_fractions[current_p_idx]
    orig_frac = get(fractions_curves[current_p_idx], original_labels[j], 0)
    d[new_label] = get(d, new_label, 0) + orig_frac
end

fig = basins_fractions_plot(joint_fractions, prange)
GLMakie.save(desktop("clustered_kuramoto_recurrences_by_std_and_ω1.png"), fig)

# IDea for "whether special node is in sync state":
# histogram approach; then one dimension is synchronisity, like order parmater
# the other approach is deviaiton of ω of special node from mean ω.
