Sys.islinux() && !haskey(ENV, "LAUNCH_PARALLEL") && include("launch_remote.jl")
using Revise
using Distributed

@everywhere begin
    using Revise
    using BetaZero
    include("representation_lightdark.jl")
    include("plot_lightdark.jl")
end
