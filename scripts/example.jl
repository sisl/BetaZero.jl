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
                        belief_reward=minex_belief_reward,
                        collect_metrics=true,
                        verbose=true,
                        accuracy_func=minex_accuracy_func)

# solver.mcts_solver.next_action = minexp_next_action # TODO: To be replace with policy head of the network.
# solver.onestep_solver.next_action = minexp_next_action # TODO: To be replace with policy head of the network.

# BetaZero parameters
solver.params.n_iterations = 10
solver.params.n_data_gen = 10
solver.params.n_evaluate = 0
solver.params.n_holdout = 0

# MCTS parameters
solver.mcts_solver.n_iterations = 10 # NOTE.

# Neural network parameters
solver.nn_params.use_cnn = true
solver.nn_params.n_samples = 10_000
solver.nn_params.verbose_update_frequency = 100
solver.nn_params.learning_rate = 0.0001
solver.nn_params.λ_regularization = 0.001
solver.nn_params.batchsize = 512

# Plotting parameters
solver.plot_incremental_data_gen = true
solver.plot_incremental_holdout = true
# solver.expert_results = (expert_accuracy=[0.84, 0.037], expert_returns=[11.963, 1.617], expert_label="LAVI") # POMCPOW baseline?

policy = solve(solver, pomdp)

filename_suffix = "minex.bson"
BetaZero.save_policy(policy, "BetaZero/notebooks/policy_$filename_suffix")
BetaZero.save_solver(solver, "BetaZero/notebooks/solver_$filename_suffix")



policy = solve(solver, pomdp)
# TODO: save network