using DrWatson
@quickactivate
using LaTeXStrings
using Attractors
using OrdinaryDiffEq:Vern9
using CairoMakie
using Random


# Lebovitz, N., & Mariotti, G. (2013). Edges in models of shear flow. Journal of Fluid Mechanics, 721, 386-402. doi:10.1017/jfm.2013.38
mutable struct LMD6Parameters{M}
    k::M
    σ::M
    Re::Float64
end
function LMD6Parameters(; Re = 307.)
    α = 1.1; β = π/2; γ = 5/3; 
    c_3 = sqrt(4*β^2 + γ^2); 
    c_5 = sqrt(γ^2 + α^2); 
    c_6 = sqrt(α^2*β^2 + γ^4 + 2*γ^2*α^2 + α^4 + 3/4*β^2*γ^2); 
    k = [   β^2; 
            5*β^2+ γ^2;
            c_3^2; 
            α^2 + 4*β^2; 
            α^2 + β^2 + γ^2; 
            (α^2 + β^2) + (γ^2*(4*c_5^4 + β^2*(4*α^2+γ^2)))/c_6^2]
    σ = [β*γ/c_3; 
         γ^2/c_5; 
         α^2/c_5; 
         γ*α*β/(2*c_6); 
         β^2*(4*α^2+5*γ^2)*α/(2*c_3*c_5*c_6)
         (β^2 - α^2 - γ^2)*γ^2*α/(2*c_3*c_5*c_6)
         γ^2*β^2*α/(4*c_3*c_5*c_6)]
    return LMD6Parameters(k, σ, Re)
end

function LMD6!(du, u, p, t)
    (; k, σ, Re) = p
    du[1] = -u[1]*k[1]/Re -σ[1]*u[2]*u[3]; 
    du[2] = -u[2]*k[2]/Re + σ[1]*u[3] + σ[1]*u[1]*u[3] - σ[2]*u[4]*u[5]; 
    du[3] = -u[3]*k[3]/Re - (σ[5] + σ[6])*u[5]*u[6]; 
    du[4] = -u[4]*k[4]/Re - σ[4]*u[6] + σ[3]*u[2]*u[5] - σ[4]*u[1]*u[6]; 
    du[5] = -u[5]*k[5]/Re + (σ[2] - σ[3])*u[2]*u[4] + (σ[5] - σ[7])*u[3]*u[6]; 
    du[6] = -u[6]*k[6]/Re + σ[4]*u[4] + (σ[6] + σ[7])*u[3]*u[5] + σ[4]*u[1]*u[4] 
end

function compute_LM(di::Dict)
    @unpack res, Re = di
    p = LMD6Parameters(; Re = Re)
    ds = ContinuousDynamicalSystem(LMD6!, zeros(6), p, (J,z0, p, n) -> nothing)
    diffeq = (alg = Vern9(), reltol = 1e-9, maxiters = 1e8)
    yg = range(-5, 5; length = 10001)
    grid = ntuple(x -> yg, dimension(ds))
    mapper = AttractorsViaRecurrences(ds, grid; sparse = true, Δt = .01,   
        mx_chk_fnd_att = 300, stop_at_Δt = true,
        mx_chk_loc_att = 100, safety_counter_max = Int(1e7), diffeq)
    u0(x,y) = [x, -0.0511, -0.0391, 0.0016, y, 0.126]
    y1r = range(-1, 1, length = res)
    y2r = range(-1, 1, length = res)
    ics = [ u0(y1,y2) for y1 in y1r, y2 in y2r]
    bsn = [ mapper(u) for u in ics]
    grid = (y1r,y2r)
    return @strdict(bsn, grid, Re, res)
end

function continuation_LMD6()
    Re_range = range(290,420, length = 50)
    p = LMD6Parameters(; Re = 307.)
    ds = ContinuousDynamicalSystem(LMD6!, zeros(6), p, (J,z0, p, n) -> nothing)
    diffeq = (alg = Vern9(), reltol = 1e-9, maxiters = 1e8)
    yg = range(-5, 5; length = 1501)
    grid = ntuple(x -> yg, 6)
    mapper = AttractorsViaRecurrences(ds, grid; sparse = true, Δt = 1.,   
        mx_chk_fnd_att = 4000, stop_at_Δt = true, store_once_per_cell = true,
        mx_chk_loc_att = 100, mx_chk_safety = Int(1e7), diffeq)
    pidx = :Re; spp = 4000
    sampler, = Attractors.statespace_sampler(Random.MersenneTwister(1234); min_bounds = ones(6).*(-1.), max_bounds = ones(6).*(1.))

    ## RECURENCE CONTINUATION
    continuation = RecurrencesSeedingContinuation(mapper; threshold = 0.2)
    fs, att = basins_fractions_continuation(
            continuation, Re_range, pidx, sampler;
            show_progress = true, samples_per_parameter = spp
            )
    return fs, att, Re_range
end

function print_basins(w,h,cmap, Re, res)
    params = @strdict res Re
    @time data, file = produce_or_load(
        datadir("basins"), params, compute_LM;
        prefix = "lebovitz", storepatch = false, suffix = "jld2", force = false
    )
    @unpack bsn, grid = data
    xg, yg = grid
    fig = Figure(resolution = (w, h))
    ax = Axis(fig[1,1], ylabel = L"$y$", xlabel = L"x", yticklabelsize = 30, 
            xticklabelsize = 30, 
            ylabelsize = 30, 
            xlabelsize = 30, 
            xticklabelfont = "cmr10", 
            yticklabelfont = "cmr10")
    if isnothing(cmap)
        heatmap!(ax, xg, yg, bsn, rasterize = 1)
    else
        heatmap!(ax, xg, yg, bsn, rasterize = 1, colormap = cmap)
    end
    save(string(projectdir(), "/plots/lebovitz_",res,".png"),fig)
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


# print_fig(600,600, nothing, 307., 450)
f,a,r = continuation_LMD6()
plot_filled_curves(f,r,"tst.png")
