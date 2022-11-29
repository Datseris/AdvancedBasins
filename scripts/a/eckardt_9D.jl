using DrWatson
@quickactivate
using LaTeXStrings
using Attractors
# using DynamicalSystems
using OrdinaryDiffEq:Vern9
using CairoMakie
using Random
using ProgressMeter

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


function continuation_E9D()
    Re_range = range(300,500, length = 50)
    p = E9DParameters(; Re = 200.)
    ds = ContinuousDynamicalSystem(E9D!, zeros(9), p, (J,z0, p, n) -> nothing)
    diffeq = (alg = Vern9(), reltol = 1e-9, maxiters = 1e8)
    yg = range(-15, 15; length = 10001)
    grid = ntuple(x -> yg, 9)
    mapper = AttractorsViaRecurrences(ds, grid; sparse = true, Δt = .01,   
        mx_chk_fnd_att = 2000, stop_at_Δt = false, store_once_per_cell = true,
        mx_chk_loc_att = 100, mx_chk_safety = Int(1e7), diffeq, show_progress = true)
    pidx = :Re; spp = 1000
    sampler, = Attractors.statespace_sampler(Random.MersenneTwister(1234); min_bounds = ones(9).*(-1.), max_bounds = ones(9).*(1.))

    ## RECURENCE CONTINUATION
    continuation = RecurrencesSeedingContinuation(mapper; threshold = 0.2)
    fs, att = basins_fractions_continuation(
            continuation, Re_range, pidx, sampler;
            show_progress = true, samples_per_parameter = spp
            )
    return fs, att, Re_range
end

function plot_filled_curves(fractions, prms, figurename)
    ff = deepcopy(fractions)
# We rearrange the fractions and we sweep under the carpet the attractors with 
# less the 5% of basin fraction. They are merged under the label -1
    for (n,e) in enumerate(fractions)
        vh = Dict();
        d = sort(e; byvalue = true)
        v = collect(values(d))
        k = collect(keys(d))
        ind = findall(v .> 0.05)
        for i in ind; push!(vh, k[i] => v[i]); end        
        ind = findall(v .<= 0.05)
        if length(ind) > 0 
            try 
                if vh[-1] > 0.05
                    vh[-1] += sum(v[ind])
                else 
                    vh[-1] = sum(v[ind])
                end
            catch 
                push!(vh, -1 => sum(v[ind]))
            end
        end
        # push!(ff, vh)
        ff[n] = vh
    end
    fractions_curves = ff

    ukeys = Attractors.unique_keys(fractions_curves)
    # ps = 1:length(fractions_curves)
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
    ax = Axis(fig[1,1])
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

f,a,r = continuation_E9D()
plot_filled_curves(f,r,"tst.png")
