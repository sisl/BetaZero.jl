include("launch_rocksample.jl")
include("solver_rocksample.jl")

filename_suffix = "rocksample"

policy = solve(solver, pomdp)
BetaZero.save_policy(policy, "policy_$filename_suffix.bson")
BetaZero.save_solver(solver, "solver_$filename_suffix.bson")
