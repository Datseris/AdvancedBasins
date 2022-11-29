using CairoMakie
import Downloads

try
    Downloads.download(
        "https://raw.githubusercontent.com/Datseris/plottheme/main/plottheme.jl",
        joinpath(@__DIR__, "plottheme.jl")
    )
catch
end

include("plottheme.jl")