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
featurizer = featurizer_kuramoto

# use API to cluster attractors
clust_config = GroupViaClustering(min_neighbors = 10)
joint_fractions = aggregate_attractor_fractions(
    fractions_curves, attractors_info, featurizer, clust_config
)

fig = basins_fractions_plot(joint_fractions, prange)
GLMakie.save(desktop("clustered_kuramoto_recurrences_by_std_and_ω1.png"), fig)

# IDea for "whether special node is in sync state":
# histogram approach; then one dimension is synchronisity, like order parmater
# the other approach is deviaiton of ω of special node from mean ω.
