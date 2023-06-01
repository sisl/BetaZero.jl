using Random
include(joinpath(@__DIR__, "..", "launch_lightdark.jl"))

USE_STD = [false] # already ran results with [μ, σ] ("solver_lightdark_maxqn_seedX.bson")

global solver, policy, filename_suffix

for use_std in USE_STD
    suffix = use_std ? "mean_std" : "mean_only"
    for seed in 1:3
        global solver, policy, filename_suffix
        Random.seed!(seed)

        include(joinpath(@__DIR__, "..", "solver_lightdark.jl"))
        !use_std && @warn "Including STD in belief rep. is done manually in BetaZero.input_representation"

        @info use_std, seed

        policy = solve(solver, pomdp)
        filename_suffix = "lightdark_$(suffix)_seed$seed.bson"
        BetaZero.save_policy(policy, "policy_$filename_suffix")
        BetaZero.save_solver(solver, "solver_$filename_suffix")
    end
end
