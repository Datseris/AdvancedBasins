using DrWatson
@quickactivate
using Attractors, OrdinaryDiffEq, GLMakie, Random
include(srcdir("vis", "basins_plotting.jl"))

diffeq = (alg = Vern9(), reltol = 1e-9, abstol = 1e-9, maxiters = 1e12)
# from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8648447
# Two coexisting non-symmetric chaotic attractors
function xingrong_rule(u, p, t)
    x, y, z = u
    α, β, γ = p
    dx = x*(y - 1) - β*z
    dy = α*(1 - x^2) - y
    dz = x - γ*z
    return SVector(dx, dy, dz)
end
@inbounds function xingrong_rule!(du, u, p, t)
    x, y, z = u
    α, β, γ = p
    du[1] = x*(y - 1) - β*z
    du[2] = α*(1 - x^2) - y
    du[3] = x - γ*z
    return
end

function xingrong(u0 = [-1.5, -1.5, 0.75]; α = 20, γ = 1.2, β = 2.62)
    p = [α, β, γ]
    return ContinuousDynamicalSystem(xingrong_rule, u0, p)
end

u_chaos = [-1.5, -1.5, 0.75]
u_limit = [-0.33,-1.5, 0.75]
u0s = [u_chaos, u_limit]
ds = xingrong()

fig = Figure()
ax = Axis3(fig[1,1])
for u0 in u0s
    tr = trajectory(ds, 10000.0, u0; Ttr = 1000)
    lines!(ax, columns(tr)...)
end
display(fig)

xg = range(-5, 5; length = 401)
yg = range(-15, 15; length = 401)
zg = range(-2, 2; length = 401)
grid = (xg, yg, zg)

mapper = AttractorsViaRecurrences(ds, grid;
    mx_chk_fnd_att = 1000,
    mx_chk_loc_att = 2000,
    mx_chk_att = 10,
    mx_chk_lost = 1000,
    safety_counter_max = 1e8,
    Δt = 0.1,
    diffeq,
)

sampler, = statespace_sampler(Random.MersenneTwister(1234);
    min_bounds = minimum.(grid), max_bounds = maximum.(grid)
)

fs = basins_fractions(mapper, sampler; N = 10000)
# Dict{Int64, Float64} with 2 entries:
#   2 => 0.1262
#   1 => 0.8738

# %% coupling the systems via diffusive coupling
# out of place form
mutable struct DiffusivelyCoupledParams{P,T,A,F}
    p1::P
    p2::P
    k1::T
    k2::T
    a1::A
    a2::A
    i::Int
    f::F
end

function diffusively_couple(ds::ContinuousDynamicalSystem{true}, i::Int, k1::Real;
    p1 = ds.p, k2 = -k1, p2 = ds.p)

    p = DiffusivelyCoupledParams(p1, p2, k1, k2, i, ds.f)
    D = dimension(ds)

    function diffusive_rule_iip!(du, u, p, t)
        (; p1, p2, k1, k2, i, f) = p
        u1, u2 = view(u, 1:D), view(u, D+1:2D)
        du1, du2 = view(du, 1:D), view(du, D+1:2D)
        f(du1, u1, p1, t)
        f(du2, u2, p2, t)
        du1[i] += k1*(u1[i] - u2[i])
        du2[i] += k2*(u1[i] - u2[i])
        return
    end
    u = vcat(get_state(ds), get_state(ds))
    return ContinuousDynamicalSystem(diffusive_rule_iip!, u, p)
end

using Attractors.DelayEmbeddings.StaticArrays: setindex

function diffusively_couple(ds::ContinuousDynamicalSystem{false}, i::Int, k1::Real;
    p1 = ds.p, k2 = -k1, p2 = ds.p)

    D = dimension(ds)
    a1 = SVector{D, Int}(1:D)
    a2 = SVector{D, Int}(D+1:2D)
    p = DiffusivelyCoupledParams(p1, p2, k1, k2, a1, a2, i, ds.f)

    u = vcat(get_state(ds), get_state(ds))
    return ContinuousDynamicalSystem(diffusive_rule_oop, u, p)
end
@inbounds function diffusive_rule_oop(u, p, t)
    (; p1, p2, k1, k2, a1, a2, i, f) = p
    u1, u2 = u[a1], u[a2]
    du1 = f(u1, p1, t)
    du2 = f(u2, p2, t)
    diff = u1[i] - u2[i]
    du1 = setindex(du1, du1[i] + k1*diff, i)
    du2 = setindex(du2, du2[i] + k2*diff, i)
    return vcat(du1, du2)
end

k = 0.1
dcds = diffusively_couple(ds, 2, k)
using ChaosTools: lyapunov

fig = Figure(resolution = (1000, 500))
ax = Axis3(fig[1,1])
ax2 = Axis3(fig[1,2])
λs = []
for u0 in u0s
    u = vcat(u0, u0) .+ 1e-6randn(6)
    tr = trajectory(dcds, 20000.0, u; Ttr = 100.0)
    x, y, z, x2, y2, z2 = columns(tr)
    λ = lyapunov(dcds, 20000.0; u0 = u, Ttr = 100.0)
    push!(λs, λ)
    lines!(ax, x, y, z)
    lines!(ax2, x2, y2, z2)
end
figuretitle!(fig, "k = $(k), λ blue = $(round(λs[1];sigdigits = 3)), λ orang = $(round(λs[2];sigdigits = 3)), ")
display(fig)
