Sys.islinux() && include("launch_remote.jl")
using Revise
using Distributed

@everywhere begin
    using BetaZero
    # include("minex_pomdp.jl")
    # include("minex_representation.jl")
    include("simple_minex_pomdp.jl")
end

solver = BetaZeroSolver(pomdp=pomdp,
                        updater=up,
                        belief_reward=simple_minex_belief_reward,
                        collect_metrics=true,
                        verbose=true,
                        accuracy_func=simple_minex_accuracy_func)

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

policy = solve(solver, pomdp)

filename_suffix = "minex.bson"
BetaZero.save_policy(policy, "BetaZero/notebooks/policy_$filename_suffix")
BetaZero.save_solver(solver, "BetaZero/notebooks/solver_$filename_suffix")
