# Climate bistable toy model from Gelbrecht et al. 2021
# Should yield Fig. 3 of the paper
function lorenz96_ebm_gelbrecht_rule!(dx, x, p, t)
    N = length(x) - 1 # number of grid points of Lorenz 96
    T = x[end]
    a₀ = 0.5
    a₁ = 0.4
    S = p[1] # Solar constant, by default 8.0
    F = 8.0
    Tbar = 270.0
    ΔT = 60.0
    α = 2.0
    β = 1.0
    σ = 1/180
    E = 0.5*sum(x[n]^2 for n in 1:N)
    𝓔 = E/N
    forcing = F*(1 + β*(T - Tbar)/ΔT)
    # 3 edge cases
    @inbounds dx[1] = (x[2] - x[N - 1]) * x[N] - x[1] + forcing
    @inbounds dx[2] = (x[3] - x[N]) * x[1] - x[2] + forcing
    @inbounds dx[N] = (x[1] - x[N - 2]) * x[N - 1] - x[N] + forcing
    # then the general case
    for n in 3:(N - 1)
      @inbounds dx[n] = (x[n + 1] - x[n - 2]) * x[n - 1] - x[n] + forcing
    end
    # Temperature equation
    dx[end] = S*(1 - a₀ + (a₁/2)*tanh(T-Tbar)) - (σ*T)^4 - α*(𝓔/(0.6*F^(4/3)) - 1)
    return nothing
end
# function that returns a dynamical system, ala `Systems.predefined_system`.
function lorenz96_ebm_gelbrecht(; N = 32, S = 8.0)
    u0 = [rand(N)..., 230.0]
    p0 = [S] # solar constant
    ds = ContinuousDynamicalSystem(lorenz96_ebm_gelbrecht_rule!, u0, p0)
    return ds
end
# Above system, but projected to the last `P` dimensions
function lorenz96_ebm_gelbrecht_projected(; P = 6, N = 32, kwargs...)
    ds = lorenz96_ebm_gelbrecht(; N, kwargs...)
    projection = (N-P+1):(N+1)
    complete_state = zeros(N-length(projection)+1)
    pinteg = projected_integrator(ds, projection, complete_state)
    return pinteg
end

# Cell differentiation model (need citation)
function cell_differentiation_rule!(du, u, p, t)
    Kd, α, β, n = p
    sum_u = sum(u)
    @inbounds for i ∈ eachindex(du)
        C = (2*u[i]^2) / (Kd + 4*sum_u + sqrt( Kd^2 + 8*sum_u*Kd )  )
        du[i] = α + (β*C^n)/(1+C^n) - u[i]
    end
    return nothing
end

function cell_differentiation(N = 3, u0 = rand(N); α=4, β=20, n=1.5, Kd=1.0)
    p = [Kd, α, β, n]
    ds = ContinuousDynamicalSystem(cell_differentiation_rule!, u0, p)
    return ds
end

# Network of (2nd order) Kuramoto oscillators,
# from original MCBB paper.
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
function KuramotoParameters(; N, K, α = 0.1, seed = 53867481290)
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
function kuramoto_network_2ndorder(; N=10, K=6.0, kwargs...)
    p = KuramotoParameters(; N, K)
    dummyjac = (J, z0, p, n) -> nothing
    ds = ContinuousDynamicalSystem(second_order_kuramoto!, zeros(2N), p, dummyjac)
    return ds
end