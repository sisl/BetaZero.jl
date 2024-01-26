Sys.islinux() && !haskey(ENV, "LAUNCH_PARALLEL") && include("launch_remote.jl")
using Revise
using Distributed

@everywhere begin
    using Revise
    using ARDESPOT
    using BetaZero
    BetaZero.probability_vector_despot(crit, info) = ARDESPOT.probability_vector_despot(crit, info) # ! NOTE.
    include("representation_lightdark.jl")
    include("plot_lightdark.jl")
end
