# This file defines convenience functions for plotting basins fractions
include(
    joinpath(pathof(Attractors), "..", "..", "docs", "basins_plotting.jl")
)

# Specific themeing for paper
include("theme.jl")
