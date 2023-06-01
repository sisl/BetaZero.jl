include("launch_robot.jl")
include("solver_robot.jl")

filename_suffix = "robot"

policy = solve(solver, pomdp)
BetaZero.save_policy(policy, "policy_$filename_suffix.bson")
BetaZero.save_solver(solver, "solver_$filename_suffix.bson")
