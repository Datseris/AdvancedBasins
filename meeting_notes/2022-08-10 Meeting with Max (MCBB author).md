# To discuss:

1. Are there any differences with bSTAB, besides clustering by using features from different parameter values rather than features all at the same parameter?
2. How do you do the clustering? (how to find DBSCAN radius)
3. How to choose integration time?
4. Main performance bottleneck? For us it is finding optimal DBSCAN radius
5. What is the "modified" DBSCAN algorithm that you use, and why didn't you use Clustering.jl?
6. My idea for a generalized interface for `basin_fraction_continuation(mapper, ...)` function. Your method becomes one of the mappers. Probably, the same as `AttractorsViaFeaturizing`.

---

1 = the main difference is what features are chosen (bSTAB doesn't give info) and what is the distance function given to DBSCAN algorithm. In MCBB the distance function includes the parameter value.
2 = You create the distance matrix and you look at the distribution of the distances to the k-th nearest neighbor and then you order this by magnitude. Then you have a graph of a distribution of how far each neighbor is. In the DBSCAN paper they recommend using the 4-th nearest neighbor. I think this is what we do as well.

5 = a mix of using Clustering.jl and writing own code.

Brief explanation of how it works MCBB
1. Sample both initial condition distribution and parameter range
2. Integrate a lot of trajectories
3. In the distance matrix there is also a term for parameter proximity to basically favor in the distance matrix (as an input to DBSCAN) attractors that are close to each other in terms of parameter values.
4. Then we do the DBSCAN. It outputs clusters that span the whole parameter range.
5. Then to derive the freaction continiation graph, we do a sliding window over the parameter range where we just count the ratio of the basins for every window.

They made a different clustering method with histograms of feature space, which seems to work very well for oscillator networks.

proposed interface for the continuation code:
```julia
function basins_fraction_continuation(
    mapper::AttractorMapper, [match_attractors,] ps::Range; kwargs...)

end

mcbb = basins_fraction_continuation(AttractorsViaFeaturizing, match_via_clustering, ps)
```