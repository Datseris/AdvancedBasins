using DrWatson 
@quickactivate
using DynamicalSystems
using Attractors
using JLD2
using CairoMakie



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

# # Clustering with Threshold Inf (Euclidean)  and samples taken from uniform dist [-π,π].
# d = load(string(projectdir(), "/data/basins/cont_kur_mcbb_samp_pi_pi.jld2"))
# f = d["fractions"]
# plot_filled_curves(f, "kur_mcbb_pi_pi_threshold_inf.png")


# # Clustering with Threshold 1. (Eunclidean norm)  and samples taken from uniform dist [-π,π].
# d = load(string(projectdir(), "/data/basins/cont_kur_mcbb_samp_pi_pi_radius_1.jld2"))
# f = d["fractions"]
# plot_filled_curves(f, "kur_mcbb_pi_pi_threshold_1.png")


# # Clustering with Threshold 1. (Hausdorff norm)  and samples taken from uniform dist [-π,π].
# d = load(string(projectdir(), "/data/basins/cont_kur_mcbb_samp_pi_pi_radius_1_hausdorff.jld2"))
# f = d["fractions"]
# plot_filled_curves(f, "kur_mcbb_pi_pi_threshold_1_hausdorff.png")

# @load "fraction_test_continuation_kur.jld2"
# plot_filled_curves(f, "kur_mcbb_continuation_method.png")
#
# 
# Clustering with Threshold Inf (Euclidean)  and samples taken from uniform dist [-π,π].
d = load("fraction_test_continuation_kur.jld2")
f = d["f"]
K = d["K"]
plot_filled_curves(f, K,  "kur_continuation_recurrence_pi_pi.png")

d = load("fraction_test_continuation_kur_mccb_8000.jld2")
f = d["f"]
K = d["K"]
plot_filled_curves(f, K,  "kur_continuation_mcbb_pi_pi_8000.png")


d = load("fraction_test_continuation_kur_new_defs.jld2")
f = d["f"]
K = d["K"]
plot_filled_curves(f, K,  "kur_continuation_recurrence_pi_pi_new_def.png")
