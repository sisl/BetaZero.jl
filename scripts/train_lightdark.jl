include("launch_lightdark.jl")
include("solver_lightdark.jl")

load_exist = false
filename_suffix = "lightdark"

if load_exist
    policy = (BSON.load("policy_$filename_suffix.bson"))[:policy];
    solver = (BSON.load("solver_$filename_suffix.bson"))[:solver];
else
    policy = solve(solver, pomdp)
    BetaZero.save_policy(policy, "policy_$filename_suffix.bson")
    BetaZero.save_solver(solver, "solver_$filename_suffix.bson")
end

# value_and_policy_plot(pomdp, policy, solver=solver, use_pgf=false)
# Plots.savefig("value_policy_plot_$filename_suffix.png")
# display(plot_lightdark(pomdp, policy, up)) # Single episode trajectory example
