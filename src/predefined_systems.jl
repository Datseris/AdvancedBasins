using DynamicalSystemsBase
using OrdinaryDiffEq: Tsit5, Vern9
using SparseArrays
const diffeq_fast = (alg = Tsit5(), reltol = 1e-6, abstol = 1e-6)
const diffeq_slow = (alg = Vern9(), reltol = 1e-9, abstol = 1e-9)
diffeq_default = diffeq_slow # choose high accuracy by default

# Lorenz 84
function lorenz84(u = [0.1, 0.1, 0.1]; F=6.846, G=1.287, a=0.25, b=4.0)
    return CoupledODEs(lorenz84_rule, u, [F, G, a, b]; diffeq = diffeq_default)
end
@inline @inbounds function lorenz84_rule(u, p, t)
    F, G, a, b = p
    x, y, z = u
    dx = -y^2 -z^2 -a*x + a*F
    dy = x*y - y - b*x*z + G
    dz = b*x*y + x*z - z
    return SVector{3}(dx, dy, dz)
end


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
    σ = 1/180 # not to the power of 4, because we do it at the equations below
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
    ds = CoupledODEs(lorenz96_ebm_gelbrecht_rule!, u0, p0; diffeq = diffeq_default)
    return ds
end
# Above system, but projected to the last `P` dimensions
function lorenz96_ebm_gelbrecht_projected(; P = nothing, N = 32, kwargs...)
    ds = lorenz96_ebm_gelbrecht(; N, kwargs...)

    # Project in same space as the paper:
    if isnothing(P)
        function gelbrecht_fig3_projection(u)
            T = u[end]
            M = E = 0.0
            @inbounds for i in 1:N
                M += u[i]
                E += u[i]^2
            end
            M /= N
            E /= 2N
            return SVector(M, E, T)
        end

        dummy = zeros(N+1)

        function gelbrect_fig3_complete(y)
            dummy[end] = y[end]
            M = y[1]
            for i in 1:N
                dummy[i] = M + (i%3)*0.01
            end
            return dummy
        end

        pinteg = ProjectedDynamicalSystem(ds, gelbrecht_fig3_projection, gelbrect_fig3_complete)
    else
        # Projection of the last `P` Lorenz96 state variables
        P == N && return ds
        projection = (N-P+1):(N+1)
        complete_state = zeros(N-length(projection)+1)
        pinteg = ProjectedDynamicalSystem(ds, projection, complete_state)
    end

    return pinteg
end

# Cell differentiation model (need citation)
function cell_differentiation(N = 3, u0 = rand(N); α=4, β=20, n=1.5, Kd=1.0)
    p = [Kd, α, β, n]
    ds = CoupledODEs(cell_differentiation_rule!, u0, p; diffeq = diffeq_default)
    return ds
end
function cell_differentiation_rule!(du, u, p, t)
    Kd, α, β, n = p
    sum_u = sum(u)
    @inbounds for i ∈ eachindex(du)
        C = (2*u[i]^2) / (Kd + 4*sum_u + sqrt( Kd^2 + 8*sum_u*Kd )  )
        du[i] = α + (β*C^n)/(1+C^n) - u[i]
    end
    return nothing
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
    ds = CoupledODEs(second_order_kuramoto!, zeros(2N), p; diffeq = diffeq_default)
    return ds
end

# A low-dimensional model for turbulent shear flows
# Jeff Moehlis , Holger Faisst and Bruno Eckhardt
# DOI: 10.1088/1367-2630/6/1/056
# Geometry of the edge of chaos in a low-dimensional turbulent shear flow model
# Madhura Joglekar,Ulrike Feudel, and James A. Yorke
# DOI: 10.1103/PhysRevE.91.052903
mutable struct E9DParameters{M}
    k::M
    σ::M
    Re::Float64
end
function E9DParameters(; Re = 307.)
   Lx = 1.75π; Lz = 1.2π
   α = 2π/Lx; β = π/2; γ = 2π/Lz;
    Kαγ = sqrt(α^2 + γ^2);
    Kβγ = sqrt(β^2 + γ^2);
    Kαβγ = sqrt(α^2 + β^2 + γ^2)
    k = [   β^2;
            4*β^2/3+ γ^2;
            β^2+γ^2;
            (3*α^2+4*β^2)/3;
            α^2 + β^2;
            (3*α^2 + 4*β^2 + 3*γ^2)/3;
            α^2 + β^2 + γ^2;
            α^2 + β^2 + γ^2;
            9*β^2]
    σ = [-√(3/2)*β*γ/Kαβγ;  √(3/2)*β*γ/Kβγ;
         5√2*γ^2/(3√3*Kαγ); -γ^2/(√6*Kαγ); -α*β*γ/(√6*Kαγ*Kαβγ); -√(3/2)*β*γ/Kβγ; -√(3/2)*β*γ/Kβγ;
         2*α*β*γ/(√6*Kαγ*Kβγ); (β^2*(3*α^2+γ^2)-3*γ^2*(α^2+γ^2))/(√6*Kαγ*Kβγ*Kαβγ);
         -α/√6; -10*α^2/(3*√6*Kαγ); -√(3/2)*α*β*γ/(Kαγ*Kβγ); -√(3/2)*α^2*β^2/(Kαγ*Kβγ*Kαβγ); -α/√6;
         α/√6; α^2/(√6*Kαγ); -α*β*γ/(√6*Kαγ*Kαβγ); α/√6; 2*α*β*γ/(√6*Kαγ*Kβγ);
         α/√6; √(3/2)*β*γ/Kαβγ; 10*(α^2 - γ^2)/(3√6*Kαγ); -2√2*α*β*γ/(√3*Kαγ*Kβγ); α/√6; √(3/2)*β*γ/Kαβγ;
         -α/√6; (γ^2-α^2)/(√6*Kαγ); α*β*γ/(√6*Kαγ*Kβγ);
         2*α*β*γ/(√6*Kαγ*Kαβγ); γ^2*(3*α^2-β^2+3*γ^2)/(√6*Kαγ*Kβγ*Kαβγ);
        √(3/2)*β*γ/Kβγ;  -√(3/2)*β*γ/Kαβγ
        ]
    return E9DParameters(k, σ, Re)
end
function E9D!(du, u, p, t)
    (; k, σ, Re) = p
    du[1] = -u[1]*k[1]/Re + σ[1]*u[6]*u[8] + σ[2]*u[2]*u[3] + k[1]/Re;
    du[2] = -u[2]*k[2]/Re + σ[3]*u[4]*u[6] + σ[4]*u[5]*u[7] + σ[5]*u[5]*u[8] + σ[6]*u[1]*u[3] + σ[7]*u[3]*u[9];
    du[3] = -u[3]*k[3]/Re + σ[8]*(u[4]*u[7]+u[5]*u[6]) + σ[9]*u[4]*u[8];
    du[4] = -u[4]*k[4]/Re + σ[10]*u[1]*u[5] + σ[11]*u[2]*u[6] + σ[12]*u[3]*u[7] + σ[13]*u[3]*u[8] + σ[14]*u[5]*u[9];
    du[5] = -u[5]*k[5]/Re + σ[15]*u[1]*u[4] + σ[16]*u[2]*u[7] + σ[17]*u[2]*u[8] + σ[18]*u[4]*u[9] + σ[19]*u[3]*u[6];
    du[6] = -u[6]*k[6]/Re + σ[20]*u[1]*u[7] + σ[21]*u[1]*u[8] + σ[22]*u[2]*u[4]+ σ[23]*u[3]*u[5] + σ[24]*u[7]*u[9] + σ[25]*u[8]*u[9]
    du[7] = -u[7]*k[7]/Re + σ[26]*(u[1]*u[6]+u[6]*u[9]) + σ[27]*u[2]*u[5] + σ[28]*u[3]*u[4]
    du[8] = -u[8]*k[8]/Re + σ[29]*u[2]*u[5] + σ[30]*u[3]*u[4]
    du[9] = -u[9]*k[9]/Re + σ[31]*u[2]*u[3] + σ[32]*u[6]*u[8]
end
function Eckhardt_9D(Re = 337.)
    p = E9DParameters(; Re = Re)
    ds = CoupledODEs(E9D!, zeros(9), p; diffeq = diffeq_default)
    return ds
end

# Population dynamics model from Huisman, 2001
# https://www.jstor.org/stable/10.1086/319929
function competition(paperfigurelabel="2")
    p = CompetitionDynamics(paperfigurelabel)
    N = size(p.Ks, 2)
    u0 = [[0.1 for i=1:N]; [S for S in p.Ss]]
    ds = CoupledODEs(competition!, u0, p; diffeq = diffeq_default);
    return ds
end

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

#not the most optimized but runs fine
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

mutable struct CompetitionDynamics
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
    return CompetitionDynamics(rs, ms, Ss, μs, Rcoups, Ks, cs, D)
end
