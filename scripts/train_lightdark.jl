is_constrained = false
resume = false
use_despot = true # ! NOTE.

if resume
    policy = solve(solver, pomdp; surrogate=policy.surrogate, resume, plot_callback=ld_plot_callback)
else
    include("launch_lightdark.jl")
    include("solver_lightdark.jl")
    policy = solve(solver, pomdp; plot_callback=ld_plot_callback)
end

filename_suffix = "lightdark10_lets_drive_phi"
# BetaZero.save_policy(policy, "policy_$filename_suffix.bson")
# BetaZero.save_solver(solver, "solver_$filename_suffix.bson")
