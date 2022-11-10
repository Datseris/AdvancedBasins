using DrWatson
@quickactivate
using OrdinaryDiffEq:Vern9
using DynamicalSystemsBase
using Attractors
using CairoMakie

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

# using BenchmarkTools
# let
#     p = CompetitionDynamics("1")
#     N = size(p.Ks, 2)
#     u0 = [[0.1 for i=1:N]; [S for S in p.Ss]]
#     du = deepcopy(u0); u = deepcopy(u0);
#     t = 0.0
#     @btime competition!($du, $u, $p, $t)
# end

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

p = CompetitionDynamics("1")

N = size(p.Ks, 2)
u0 = [[0.1 for i=1:N]; [S for S in p.Ss]]
# N[2] = 0.1;
ds = ContinuousDynamicalSystem(competition!, u0, p, (J, z, p, t)->nothing)

T = 2000.0; Ttr = 0.0; Δt = 1.0;
tr = trajectory(ds, T; Ttr, Δt); t=Ttr:Δt:T;
plot_dynamics(p)
# save("$(plotsdir())/populationdynamics-fig1.png", fig, px_per_unit=3)

diffeq = (alg = Vern9(), maxiters=Inf);
xg = range(0, 60,length = 300);
grid = ntuple(x->xg, N+3);
mapper = AttractorsViaRecurrences(ds, grid;
        mx_chk_fnd_att = 1000,
        mx_chk_loc_att = 500,
        mx_chk_att = 100,
        # Δt = 100.,
        sparse = true,
        diffeq,
        stop_at_Δt=true,
    );

red_grid = Attractors.reduced_grid(grid, 2);
basins, atts = basins_of_attraction(mapper, red_grid);

fig, ax = plot_attractors(atts; idxs=[1,2,6])
fig

function verify_convergence_attractors(ds, atts)
    for (k, att) in atts
        u0 = att[1, :]
        reinit!()
        ds = ContinuousDynamicalSystem(competition!, u0, p, (J, z, p, t)->nothing)
        T = 2000.0; Ttr = 0.0; Δt = 1.0;
        tr = trajectory(ds, T; Ttr, Δt); t=Ttr:Δt:T;




pidx = :D; ps = 0.2:0.05:0.3
continuation = RecurrencesSeedingContinuation(mapper)
fractions_curves, attractors_info = basins_fractions_continuation(
    continuation, ps, pidx, sampler;
    show_progress = false, samples_per_parameter = 20
)
#continuate across some parameter to check for some pops extinction
fig 4a: unit 6 remains, others die off; so keep track of N[6] maybe (keep track when it is alive)

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


using CairoMakie
function plot_dynamics(p)
    fig = Figure()
    ax = Axis(fig[1,1])
    xlims!(1000, 2000)
    for i=1:N lines!(ax, t, tr[:,i]) end
    ax = Axis3(fig[2,1])
    lines!(ax, tr[:,1], tr[:,2], tr[:,3])
    fig
    return fig, t, tr, p
end