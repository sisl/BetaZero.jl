include("launch_lightdark.jl")
include("solver_lightdark.jl")

filename_suffix = "lightdark"

policy = solve(solver, pomdp)
BetaZero.save_policy(policy, "policy_$filename_suffix.bson")
BetaZero.save_solver(solver, "solver_$filename_suffix.bson")

# value_and_policy_plot(pomdp, policy)
# savefig("value_policy_plot_$filename_suffix.png")
# display(plot_lightdark(pomdp, policy, up)) # Single episode trajectory example
