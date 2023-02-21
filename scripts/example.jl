Sys.islinux() && include("launch_remote.jl")
using Revise
using Distributed

@everywhere begin
    using BetaZero
    include("minex_pomdp.jl")
    include("minex_representation.jl")
end

solver = BetaZeroSolver(pomdp=pomdp,
                        updater=up,
                        belief_reward=(pomdp::POMDP, b, a, bp)->mean(reward(pomdp, s, a) for s in MineralExploration.particles(b)),
                        n_iterations=25,
                        n_data_gen=100,
                        n_evaluate=0, # NOTE.
                        n_holdout=0, # NOTE.
                        collect_metrics=false,
                        verbose=true,
                        accuracy_func=minex_accuracy_func)

# solver.mcts_solver.next_action = minexp_next_action # TODO: To be replace with policy head of the network.
# solver.onestep_solver.next_action = minexp_next_action # TODO: To be replace with policy head of the network.
solver.network_params.verbose_plot_frequency = solver.network_params.training_epochs
solver.network_params.verbose_update_frequency = 100

policy = solve(solver, pomdp)
# TODO: save network