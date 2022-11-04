# Functions used with DrWatson's `produce_or_load` to produce the basins fractions
# of attractors of dynamical systems given a specific configuration struct.

"""
Dedicated configuration container for finding and continuating basins fractions
versus parameter for a given dynamical system. This is the recurrences version.
"""
struct FractionsRecurrencesConfig
    name::String
    ds::GeneralizedDynamicalSystem
    prange::AbstractVector
    pidx::Union{Integer, Symbol}
    grid::Tuple
    mapper_config::NamedTuple
    samples_per_parameter::Int
end

function fractions_produce_or_load(config::FractionsRecurrencesConfig; force = false)
    # used to obtain a hash from `config` without using "bad" fields like `ds`
    function pure_hash(config)
        (; name, prange, grid, mapper_config, samples_per_parameter) = config
        return hash((name, prange, grid, mapper_config, samples_per_parameter))
    end
    # and finally call the coolest function in the world
    output, file = produce_or_load(
        fractions_produce_or_load_f, config, datadir("basins_fractions");
        filename = pure_hash, force, prefix = config.name, tag = false,
    )
    return output
end

function fractions_produce_or_load_f(config::FractionsRecurrencesConfig)
    (; ds, prange, pidx, grid, mapper_config, samples_per_parameter, name) = config
    mapper = AttractorsViaRecurrences(ds, grid; mapper_config..., diffeq)
    sampler, = statespace_sampler(Random.MersenneTwister(1234);
        min_bounds = minimum.(grid), max_bounds = maximum.(grid)
    )
    continuation = RecurrencesSeedingContinuation(mapper; threshold = Inf)
    fractions_curves, attractors_info = basins_fractions_continuation(
        continuation, prange, pidx, sampler;
        show_progress = true, samples_per_parameter
    )
    # Notice that we do not store the system's parameters or dynamic rule.
    # Given the name of the system,
    # they are supposed to be set in stone; only one parameter is varied
    # and the rest are constants.
    return @strdict(
        fractions_curves, attractors_info, prange, samples_per_parameter, grid, name,
    )
end
