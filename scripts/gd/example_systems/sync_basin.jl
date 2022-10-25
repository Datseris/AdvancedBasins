using DrWatson
@quickactivate
using Attractors, OrdinaryDiffEq, GLMakie
using Graphs, Random

# TODO: incidence multiplication must become in place
# and all these vector access are bad and allocating...
function second_order_kuramoto!(du, u, p, t)
    (; N, α, K, incidence, P) = p
    du[1:N] .= u[1+N:2*N]
    sine_term = K .* (incidence * sin.(incidence' * u[1:N]))
    du[N+1:end] .= P .- α .* u[1+N:2*N] .- sine_term
end

mutable struct KuramotoParameters{M}
    N::Int
    α::Float64
    incidence::M
    P::Vector{Float64}
    K::Float64
end
function KuramotoParameters(; N = 10, α = 0.1, K = 5.0, seed = 53867481290)
    rng = Random.Xoshiro(seed)
    g = random_regular_graph(N, 3; rng)
    incidence = incidence_matrix(g, oriented=true)
    P = [isodd(i) ? +1.0 : -1.0 for i = 1:N]
    return KuramotoParameters(N, α, incidence, P, K)
end

N = 10
p = KuramotoParameters(; N)

ds = ContinuousDynamicalSystem(second_order_kuramoto!, zeros(2*N), p, (J, z0, p, n) -> nothing)
diffeq = (alg = Tsit5(), reltol = 1e-9, maxiters = 1e6)

uu = trajectory(ds, 1500; Δt = 0.1, diffeq)

# %% Initial version of teh mapper that projects into phases φ
# what is going on here with the projection...? What does this `length(y)` check achieve?
# Also, why are we projecting on φ, the first variable, instead of
# ω, the second variable, as they do in the MCBB paper...?
# And why initialize the random initial conditions in
# u = vcat(π .* rand(N), (rand(N) .- 0.5) .* 12)
# when the ωs (second variables) are actually just hovering around 0...?
# with stds = 0.01 or so?
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
    Ttr = 400.0
)

n = 100
labels = []
ics = []
for k = 1:n
    u = vcat(π .* rand(N), (rand(N) .- 0.5) .* 12)
    @show l = mapper(u)
    # push!(ics, ([psys.complete_state; u],l))
    push!(labels, l)
    push!(ics, u)
end

# This finds several attractors
att = mapper.bsn_nfo.attractors

fig = Figure()
ax = Axis(fig[1,1])
for (k, a) in att
    scatterlines!(ax, a[:, 1], a[:, 2])
end
display(fig)



# %% Second version of the mapper that projects into frequencies ω
projection = N+1:2N
complete = y -> vcat(π .* rand(N), y)

psys = projected_integrator(ds, projection, complete; diffeq)

yg = range(-10, 10; length = 101)
grid = ntuple(x -> yg, dimension(psys))
mapper = AttractorsViaRecurrences(psys, grid; sparse = true, Δt = 0.1,
    diffeq,
    mx_chk_fnd_att = 100,
    mx_chk_loc_att = 100,
    safety_counter_max = Int(1e5),
    Ttr = 400.0
)

n = 100
labels = []
ics = []
for k = 1:n
    u = (rand(N) .- 0.5)
    @show l = mapper(u)
    # push!(ics, ([psys.complete_state; u],l))
    push!(labels, l)
    push!(ics, u)
end

# This finds only a single attractor???
att = mapper.bsn_nfo.attractors

fig = Figure()
ax = Axis(fig[1,1])
for (k, a) in att
    scatterlines!(ax, a[:, 1], a[:, 2])
end
display(fig)


# %% Lyapunov exponents
using ChaosTools
λs = Dict{Int, Float64}()
for i in 1:n
    l = labels[i]
    haskey(λs, l) && continue
    u = ics[i]
    λ = lyapunov(ds, 10000.0; u0 = u, Ttr = 100.0)
    λs[l] = λ
    @show l, λ
end
