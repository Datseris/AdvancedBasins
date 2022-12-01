using DrWatson
@quickactivate
using LaTeXStrings
using Attractors
# using DynamicalSystems
using OrdinaryDiffEq
using CairoMakie
using Random
using ProgressMeter
using JLD2

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

function continuation_E9D(Re_range)
    p = E9DParameters(; Re = 337.)
    ds = ContinuousDynamicalSystem(E9D!, zeros(9), p, (J,z0, p, n) -> nothing)
    diffeq = (alg = Vern9(), reltol = 1e-9, maxiters = 1e8)
    yg = range(-2, 2; length = 1001)
    grid = ntuple(x -> yg, 9)
    mapper = AttractorsViaRecurrences(ds, grid; sparse = true, Δt = 1.,   
        mx_chk_fnd_att = 1000, stop_at_Δt = true, store_once_per_cell = true,
        mx_chk_loc_att = 1000, mx_chk_safety = Int(1e7), show_progress = true,
        mx_chk_att = 10, diffeq)
    pidx = :Re; spp = 10000
    sampler, = Attractors.statespace_sampler(Random.MersenneTwister(1234); min_bounds = ones(9).*(-1.), max_bounds = ones(9).*(1.))

    ## RECURENCE CONTINUATION
    continuation = RecurrencesSeedingContinuation(mapper; threshold = Inf)
    fs, att = basins_fractions_continuation(
            continuation, Re_range, pidx, sampler;
            show_progress = true, samples_per_parameter = spp
            )
    return fs, att, Re_range
end

function continuation_projected_E9D()
    Re_range = range(280,480, length = 20)
    p = E9DParameters(; Re = 350.)
    ds = ContinuousDynamicalSystem(E9D!, zeros(9), p, (J,z0, p, n) -> nothing)
    diffeq = (alg = Vern9(), reltol = 1e-9, maxiters = 1e8)
    _complete(y) = (length(y) == 2) ? zeros(9) : y; 
    _proj_state(y) = [y[1], y[5]]
    psys = projected_integrator(ds, _proj_state, _complete; diffeq)
    yg = range(-5, 5; length = 10001)
    grid = ntuple(x -> yg, 2)
    mapper = AttractorsViaRecurrences(psys, grid; sparse = true, Δt = .1,   
        mx_chk_fnd_att = 10000, stop_at_Δt = false, store_once_per_cell = true,
        mx_chk_loc_att = 10000, mx_chk_safety = Int(1e7), show_progress = true, mx_chk_att = 30)
    pidx = :Re; spp = 5000
    sampler, = Attractors.statespace_sampler(Random.MersenneTwister(1234); min_bounds = ones(9).*(-1.), max_bounds = ones(9).*(1.))

    ## RECURENCE CONTINUATION
    continuation = RecurrencesSeedingContinuation(mapper; threshold = 0.1)
    fs, att = basins_fractions_continuation(
            continuation, Re_range, pidx, sampler;
            show_progress = true, samples_per_parameter = spp
            )
    return fs, att, Re_range
end

function plot_filled_curves(fractions, prms, figurename)
    fractions_curves = fractions
    ukeys = Attractors.unique_keys(fractions_curves)
    ps = prms
    bands = [zeros(length(ps)) for k in ukeys]
    for i in eachindex(fractions_curves)
        for (j, k) in enumerate(ukeys)
            bands[j][i] = get(fractions_curves[i], k, 0)
        end
    end
# transform to cumulative sum
    for j in 2:length(bands)
        bands[j] .+= bands[j-1]
    end

    fig = Figure(resolution = (600, 500))
    ax = Axis(fig[1,1], ylabel = "Basin Fractions", xlabel = "Re", yticklabelsize = 30,
            xticklabelsize = 30, 
            ylabelsize = 30, 
            xlabelsize = 30)
    for (j, k) in enumerate(ukeys)
        if j == 1
            l, u = 0, bands[j]
        else
            l, u = bands[j-1], bands[j]
        end
        band!(ax, ps, l, u; color = Cycled(j), label = "$k")
    end
    ylims!(ax, 0, 1)
    axislegend(ax; position = :lt)
    # display(fig)

    # save(string(projectdir(), "/plots/a/", figurename),fig)
    save(figurename,fig)
# Makie.save("lorenz84_fracs.png", fig; px_per_unit = 4)
end

function plot_bif(arange, att)
    a = arange; 
    diffeq = (alg = Vern9(), reltol = 1e-9, maxiters = 1e8)
    s = Vector{Int32}[]
    for e in att 
        push!(s, collect(keys(e)))
    end
    s = unique(vcat(s...))
    ptlst = Vector{Vector{Vector{Float64}}}(undef,length(s))
    for k in 1:length(s); ptlst[k] = []; end
    for (k,el) in enumerate(att)
        p = E9DParameters(; Re = a[k])
        ds = ContinuousDynamicalSystem(E9D!, zeros(9), p, (J,z0, p, n) -> nothing)
        @show el
        if el ≠ 0 
            for p in el
                # set_parameter!(df, [a[k], ν])
                tra = trajectory(ds, 100, p[2][1]; Ttr = 200, diffeq)
                for y in tra
                    v = [a[k], y[1]]
                    push!(ptlst[p[1]], v)
                end
            end
        end
    end
    fig = Figure(resolution = (1300, 900))
    ax = Axis(fig[1,1], ylabel = "xn", yticklabelsize = 20, xticklabelsize = 20, ylabelsize = 20)
    for (j,p) in enumerate(ptlst)
        P = Dataset(p)
        scatter!(ax, P[:,1],P[:,2], markersize = 0.7, color = Cycled(j), rasterize = 4)
    end
    save("diag_bif_e9d.png",fig)
    return ptlst 
end


Re_range = range(300,450, length = 50)
f,a,r = continuation_E9D(Re_range)
save("eckhardt_cont_full.jld2","f",f,"a",a,"r",r)
# # @load "eckhardt_cont_projected.jld2"
plot_filled_curves(f,r,"continuation_eckhardt_9D_full.png")

# ptlst = plot_bif(Re_range ,a)

