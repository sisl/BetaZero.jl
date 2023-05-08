using Random
include(joinpath(@__DIR__, "..", "launch_lightdark.jl"))

SUFFIXES = ["maxqn", "maxn", "maxq"]
CRITERIA = [MaxQN(), MaxN(), MaxQ()]

global solver
global policy

for i in eachindex(CRITERIA)
    suffix = SUFFIXES[i]
    criterion = CRITERIA[i]
    for seed in 1:3
        Random.seed!(seed)

        include(joinpath(@__DIR__, "..", "solver_lightdark.jl"))
        solver.mcts_solver.final_criterion = criterion

        @info solver.mcts_solver.final_criterion, seed

        policy = solve(solver, pomdp)
        filename_suffix = "lightdark_$(suffix)_seed$seed.bson"
        BetaZero.save_policy(policy, "policy_$filename_suffix")
        BetaZero.save_solver(solver, "solver_$filename_suffix")

        value_and_policy_plot(pomdp, policy)
        Plots.savefig("value_policy_plot_$filename_suffix.png")
    end
end
