include("launch_minex.jl")
include("solver_minex.jl")

@everywhere filename_suffix = "mineral_exploration"

policy = solve(solver, pomdp)
BetaZero.save_policy(policy, "policy_$filename_suffix.bson")
BetaZero.save_solver(solver, "solver_$filename_suffix.bson")
