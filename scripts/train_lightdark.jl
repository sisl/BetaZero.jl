resume = false
if resume
    policy = solve(solver, pomdp; surrogate=policy.surrogate, resume)
else
    include("launch_lightdark.jl")
    include("solver_lightdark.jl")
    policy = solve(solver, pomdp)
end

filename_suffix = "lightdark_testing_safety"
BetaZero.save_policy(policy, "policy_$filename_suffix.bson")
BetaZero.save_solver(solver, "solver_$filename_suffix.bson")

# value_and_policy_plot(pomdp, policy)
# savefig("value_policy_plot_$filename_suffix.png")
# display(plot_lightdark(pomdp, policy, up)) # Single episode trajectory example
