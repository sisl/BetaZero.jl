using Pkg

packages = [
    PackageSpec(url=joinpath(@__DIR__, "submodules", "MCTS")),
    PackageSpec(url=joinpath(@__DIR__, "submodules", "RemoteJobs")),
    PackageSpec(url=joinpath(@__DIR__, "submodules", "ParticleBeliefs")),
    PackageSpec(url=joinpath(@__DIR__)), # BetaZero.jl
]

if !isempty(ARGS) && ARGS[1] == "--models"
    @info "Installing POMDP models..."
    pomdp_models = [
        PackageSpec(url=joinpath(@__DIR__, "submodules", "LightDark")),
        PackageSpec(url="https://github.com/sisl/MineralExploration"),
        PackageSpec(url=joinpath(@__DIR__, "submodules", "MinEx")),

    ]
    push!(packages, pomdp_models...)
end

# Run dev altogether
# This is important that it's run together so there
# are no "expected pacakge X to be registered" errors.
Pkg.develop(packages)
