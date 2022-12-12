# using GLMakie
import Downloads

path = joinpath(@__DIR__, "plottheme.jl")
# Set default colorscheme if you'd like, see `plottheme.jl`
ENV["COLORSCHEME"] = "JuliaDynamics"

try
    Downloads.download(
        "https://raw.githubusercontent.com/Datseris/plottheme/main/plottheme.jl",
        path
    )
catch
end
if isfile(path)
    include(path)
end
