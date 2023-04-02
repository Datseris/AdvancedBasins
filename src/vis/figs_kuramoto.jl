
# We rearrange the fractions and we sweep under the carpet the attractors with
# less the 4% of basin fraction. They are merged under the label -1
function aggregate_small_fractions(fractions; threshold = 0.04)
    ff = deepcopy(fractions)
    for (n,e) in enumerate(fractions)
        vh = Dict();
        d = sort(e; byvalue = true)
        v = collect(values(d))
        k = collect(keys(d))
        ind = findall(v .> threshold)
        for i in ind; push!(vh, k[i] => v[i]); end
        ind = findall(v .<= threshold)
        if length(ind) > 0
            try
                if vh[-1] > threshold
                    vh[-1] += sum(v[ind])
                else
                    vh[-1] = sum(v[ind])
                end
            catch
                push!(vh, -1 => sum(v[ind]))
            end
        end
        ff[n] = vh
    end
    return ff
end

function plot_filled_curves_kuramoto(fractions, prms, figurename)
    fractions_curves = deepcopy(fractions)
    labs = ["0 ≤ r < 0.2"
         "0.2 ≤ r < 0.4"
         "0.4 ≤ r < 0.6"
         "0.6 ≤ r < 0.8"
         "0.8 ≤ r < 1."]

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
    ax = Axis(fig[1,1], xlabel = "K")
    for (j, k) in enumerate(ukeys)
        if j == 1
            l, u = 0, bands[j]
        else
            l, u = bands[j-1], bands[j]
        end
        # band!(ax, ps, l, u; color = Cycled(j), label = "$k")
        band!(ax, ps, l, u; color = Cycled(j), label = labs[k])
    end
    ylims!(ax, 0, 1)
    axislegend(ax; position = :lt)
    # display(fig)

    # save(string(projectdir(), "/plots/a/", figurename),fig)
    save(figurename,fig)
# Makie.save("lorenz84_fracs.png", fig; px_per_unit = 4)
end

