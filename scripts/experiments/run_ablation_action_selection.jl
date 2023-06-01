using Random
include(joinpath(@__DIR__, "..", "launch_lightdark.jl"))

USE_PRIORITY = [false] # already ran results with prioritized action selection ("solver_lightdark_maxqn_seedX.bson")

global solver, policy, filename_suffix

for use_priority in USE_PRIORITY
    suffix = use_priority ? "prioritize" : "random"
    for seed in 1:3
        global solver, policy, filename_suffix
        Random.seed!(seed)

        include(joinpath(@__DIR__, "..", "solver_lightdark.jl"))
        solver.nn_params.use_prioritized_action_selection = use_priority

        @info solver.nn_params.use_prioritized_action_selection, seed

        policy = solve(solver, pomdp)
        filename_suffix = "lightdark_$(suffix)_seed$seed.bson"
        BetaZero.save_policy(policy, "policy_$filename_suffix")
        BetaZero.save_solver(solver, "solver_$filename_suffix")
    end
end
