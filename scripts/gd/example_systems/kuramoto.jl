using DrWatson
@quickactivate
using Attractors, OrdinaryDiffEq, GLMakie, Random
using SparseArrays: sparse
using Graphs: random_regular_graph, incidence_matrix
include(srcdir("vis", "basins_plotting.jl"))

# Actually fast Kuramoto code. Comparison of performance:
# 7.900 ns (0 allocations: 0 bytes) # current
# 291.144 ns (5 allocations: 784 bytes) # previous, and already improvement of Alex's original
mutable struct KuramotoParameters{M}
    N::Int
    α::Float64
    Δ::M
    ΔT::M
    P::Vector{Float64}
    K::Float64
    # Both of these are dummies
    x::Vector{Float64}
    y::Vector{Float64}
end
function KuramotoParameters(; N = 10, α = 0.1, K = 6.0, seed = 53867481290)
    rng = Random.Xoshiro(seed)
    g = random_regular_graph(N, 3; rng)
    Δ = incidence_matrix(g, oriented=true)
    P = [isodd(i) ? +1.0 : -1.0 for i = 1:N]
    x = Δ' * zeros(N)
    y = zeros(N)
    ΔT = sparse(Matrix(Δ'))
    return KuramotoParameters(N, α, Δ, ΔT, P, K, x, y)
end
using LinearAlgebra: mul!
function second_order_kuramoto!(du, u, p, t)
    (; N, α, K, Δ, ΔT, P, x, y) = p
    φs = view(u, 1:N)
    ωs = view(u, N+1:2N)
    dφs = view(du, 1:N)
    dωs = view(du, N+1:2N)
    dφs .= ωs
    mul!(x, ΔT, φs)
    x .= sin.(x)
    mul!(y, Δ, x)
    y .*= K
    # the full sine term is y now.
    @. dωs = P - α*ωs - y
    return nothing
end

N = 10
K = 4.0
# for K < 1 you should find one or two attractors (unsynch).
# for 4 < K < 7 : zillions of attractors
# K > 9 one attractor (synchronized).
p = KuramotoParameters(; N, K)

ds = ContinuousDynamicalSystem(second_order_kuramoto!, zeros(2N), p, (J, z0, p, n) -> nothing)
diffeq = (alg = Tsit5(), reltol = 1e-9, maxiters = Inf)

uu = trajectory(ds, 1500; Δt = 0.1, diffeq)

recurrence_kwargs = (;
    mx_chk_fnd_att = 2000,
    mx_chk_loc_att = 5000,
    Ttr = 100.0,
    mx_chk_safety = Int(1e5),
    diffeq,
)

# %% Mapper that projects into frequencies ω
projection = N+1:2N
complete = y -> vcat(π .* rand(N), y)

psys = projected_integrator(ds, projection, complete; diffeq)

yg = range(-17, 17; length = 101)
grid = ntuple(x -> yg, dimension(psys))
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
psys = projected_integrator(ds, projection, complete; diffeq)
prange = range(0, 10; length = 11)
pidx = :K

mapper = AttractorsViaRecurrences(psys, grid; Δt = 0.1, recurrence_kwargs...)

continuation = RecurrencesSeedingContinuation(mapper; threshold = Inf)

fractions_curves, attractors_info = basins_fractions_continuation(
    continuation, prange, pidx;
    show_progress = true, samples_per_parameter = 10
)

fig = basins_fractions_plot(fractions_curves, prange)
display(fig)

# %% Aggregate attractors by clustering
function featurizer_kuramoto(A, t)
    ωs = A[end]
    x = std(ωs)
    return [x]
end
