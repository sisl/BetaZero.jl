using Pkg
Pkg.activate(".")
"""
Extras needed to run BetaZero
"""

@info "Installing POMDP models and tools..."
packages = [
    PackageSpec(url=joinpath(@__DIR__, "submodules", "LightDark")),
    PackageSpec(url=joinpath(@__DIR__, "submodules", "MineralExploration")),
    PackageSpec(url=joinpath(@__DIR__, "submodules", "MinEx")),
    PackageSpec(url=joinpath(@__DIR__, "submodules", "RemoteJobs")),
    PackageSpec(url=joinpath(@__DIR__, "submodules", "ParticleBeliefs")),
]

# Run dev altogether
# This is important that it's run together so there
# are no "expected pacakge X to be registered" errors.
Pkg.develop(packages)
Pkg.instantiate()