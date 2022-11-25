using DrWatson
@quickactivate
# using DynamicalSystems
using Attractors
using StaticArrays
using Random

 function new_henon(x, p, n)
    return SVector{2}(p[1] - x[1]^2 - (1 - p[2])*x[2],  x[1])
end



function _get_continuation_henon(arange)
    # @unpack res, a, ν = d
    ν = 0.01
    u0 = [0., 0.6]
    df = DiscreteDynamicalSystem(new_henon, u0, [1.,ν]) 
    # df = Systems.henon(u0; a, b)
    xg = yg = range(-25, 25, length = 5001)
    grid = (xg, yg)
    mapper = Attractors.AttractorsViaRecurrences(df, grid; sparse = true,
            mx_chk_lost = 10, 
            mx_chk_fnd_att = 3000, 
            mx_chk_loc_att = 3000, 
            mx_chk_att = 2)

    pidx = 1; spp = 10000
    sampler, = statespace_sampler(Random.MersenneTwister(1234); min_bounds = [-2., -2.], max_bounds = [2., 2.])

    ## RECURENCE CONTINUATION
    continuation = RecurrencesSeedingContinuation(mapper;
        threshold = 0.01
    )
    fs, att = basins_fractions_continuation(
            continuation, arange, pidx, sampler;
            show_progress = true, samples_per_parameter = spp
            )


    return fs,att
end

function plot_bif(arange, att)
    a = arange; ν = 0.01
    s = Vector{Int32}[]
    for e in att 
        push!(s, collect(keys(e)))
    end
    s = unique(vcat(s...))
    ptlst = Vector{Vector{Vector{Float64}}}(undef,length(s))
    for k in 1:length(s); ptlst[k] = []; end
    u0 = [0., 0.6]
    df = DiscreteDynamicalSystem(new_henon, u0, [a[1],ν]) 
    for (k,el) in enumerate(att)
        @show el
        if el ≠ 0 
            for p in el
                set_parameter!(df, [a[k], ν])
                tra = trajectory(df, 20, p[2][1]; Ttr = 200000)
                for y in tra
                    v = [a[k], y[1]]
                    push!(ptlst[p[1]], v)
                end
            end
        end
    end
    return ptlst 
end

arange = range(0.0, 3.98, length = 10000)
f,a = _get_continuation_henon(arange)
ptlst = plot_bif(arange, a)

fig = Figure(resolution = (1524, 568))
ax = Axis(fig[1,1], ylabel = "xn", yticklabelsize = 20, xticklabelsize = 20, ylabelsize = 20)
for (j,p) in enumerate(ptlst)
    P = Dataset(p)
    scatter!(ax, P[:,1],P[:,2], markersize = 0.7, color = Cycled(j), rasterize = 4)
end

save("diag_bif_henon.png",fig)
