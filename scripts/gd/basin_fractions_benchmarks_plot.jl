using DrWatson
@quickactivate
using CairoMakie, GLMakie, DataFrames
include(srcdir("vis", "theme.jl"))

# Load all benchmarks into a dataframe:
results = collect_results(datadir("comparison_benchmarks", "mappers"))
sort!(results, ["system", "method", "N"])
results_by_system = groupby(results, "system")
systems = only.(keys(results_by_system)) # only gives us the elements in nice vector format
methods = string.(sort!(unique(results[:, "method"])))

fig, axs = subplotgrid(2, length(systems);
sharex = true, titles = systems, resolution = (1000, 600))

legend_elements = []
display(fig)

for (i, (system, benchmarks)) in enumerate(zip(systems, results_by_system))
    ax_times = axs[1,i]
    ax_alloc = axs[2,i]
    grouped = groupby(benchmarks, "method")
    for (j, x) in enumerate(grouped)
        # notice that we drop the "method" column
        Ns, time_vs_N, allocs_vs_N = eachcol(x[:, ["N", "time", "allocs"]])
        c = Cycled(j)
        m = Cycled(j)
        kwargs = (color = c, marker = m, markersize = 20, linewidth = 2,)
        scatterlines!(ax_alloc, log2.(Ns), log2.(allocs_vs_N); kwargs...)
        ele = scatterlines!(ax_times, log2.(Ns), log2.(time_vs_N); kwargs...)
        if i == 1
            push!(legend_elements, ele)
            ax_times.ylabel = "log₂(time [sec])"
            ax_alloc.ylabel = "log₂(memory [byte])"
        end
        ax_alloc.xlabel = "log₂(# i.c.)"
    end
    # add titles or whatever
end

# add legend
Legend(fig[end+1, :], legend_elements, methods;
nbanks=length(methods)÷2, tellheight=true, tellwidth = false)
display(fig)
wsave(plotsdir("benchmarks", "mappers.png"), fig)