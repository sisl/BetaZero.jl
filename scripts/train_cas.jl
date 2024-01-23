is_constrained = true
resume = false
if resume
    policy = solve(solver, pomdp; surrogate=policy.surrogate, resume, plot_callback=cas_plot_callback)
else
    include("launch_cas.jl")
    include("solver_cas.jl")
    policy = solve(solver, pomdp; plot_callback=cas_plot_callback)
end

filename_suffix = "cas_alert_$(is_constrained ? "cbz" : "bz")"
BetaZero.save_policy(policy, "policy_$filename_suffix.bson")
BetaZero.save_solver(solver, "solver_$filename_suffix.bson")

#=
solver.mcts_solver.final_criterion = MCTS.MaxZQN(zq=solver.mcts_solver.final_criterion.zq, zn=solver.mcts_solver.final_criterion.zn)
policy = solve_planner!(solver, policy.surrogate)
using POMDPTools
for (s, a, r, sp, t) in stepthrough(pomdp, policy, up, b0, "s,a,r,sp,t", max_steps=20)
    @show t, a, r
end
=#
