using MineralExploration
using POMDPs
using HDF5

include("utils.jl")

const DEFAULT_N = 10_000

function generate_states(n=DEFAULT_N; grid_dims=(32,32,1))
    pomdp = MineralExplorationPOMDP(grid_dim=grid_dims, high_fidelity_dim=grid_dims)
    initialize_data!(pomdp, 0)
    ds0 = initialstate_distribution(pomdp)
    S = rand(ds0, n)
    ore_maps = [Float32.(s.ore_map[:,:,1]) for s in S]
    mat = Array{Float32}(undef, size(ore_maps[1])..., length(ore_maps))
    for i in eachindex(ore_maps)
        mat[:,:,i] = ore_maps[i]
    end
    return mat::Array{Float32}
end

function generate_and_save_states(n=DEFAULT_N; filename="data.h5", kwargs...)
    mat = generate_states(n; kwargs...)
    save_states(mat, filename)
    return mat
end
