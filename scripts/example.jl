Sys.islinux() && include("launch_remote.jl")
using Revise
using Distributed

@everywhere begin
    using BetaZero
    include("minex_pomdp.jl")
    include("minex_representation.jl")
end

solver = BetaZeroSolver(updater=up,
                        belief_reward=(pomdp::POMDP, b, a, bp)->mean(reward(pomdp, s, a) for s in particles(b)),
                        collect_metrics=true,
                        verbose=true,
                        accuracy_func=minex_accuracy_func)
solver.mcts_solver.next_action = minexp_next_action # TODO: To be replace with policy head of the network.
solver.network_params.input_size = size(BetaZero.input_representation(b0)) # TODO: Automatically infer from initial belief
solver.network_params.verbose_plot_frequency = Inf # disable plotting on bend

policy = solve(solver, pomdp)
