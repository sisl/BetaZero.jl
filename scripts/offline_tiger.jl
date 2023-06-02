Sys.islinux() && !haskey(ENV, "LAUNCH_PARALLEL") && include("launch_remote.jl")
using Revise
using Distributed

@everywhere begin
    using Revise
    using BetaZero
    include("representation_tiger.jl")
end

filename_suffix = "tiger.bson"

solver = BetaZeroSolver(pomdp=pomdp,
                        updater=up,
                        belief_reward=tiger_belief_reward,
                        params=BetaZeroParameters(
                            n_iterations=1000,
                            n_data_gen=100
                        ),
                        collect_metrics=true,
                        verbose=true,
                        plot_incremental_data_gen=true)

# Neural network
solver.nn_params.n_samples = 300
solver.nn_params.training_epochs = 500
solver.nn_params.verbose_update_frequency = 50
solver.nn_params.verbose_plot_frequency = solver.nn_params.training_epochs
solver.nn_params.batchsize = 16
solver.nn_params.learning_rate = 5e-5
solver.nn_params.λ_regularization = 1e-5
solver.nn_params.use_dropout = true
solver.nn_params.p_dropout = 0.7
solver.nn_params.use_batchnorm = true
solver.nn_params.batchnorm_momentum = 0.7f0
solver.nn_params.layer_size = 64
solver.nn_params.normalize_input = false
solver.nn_params.normalize_output = true
solver.nn_params.loss_func = BetaZero.Flux.Losses.mae

# MCTS parameters
solver.mcts_solver.n_iterations = 1000

policy = solve(solver, pomdp)
BetaZero.save_policy(policy, "data/policy_$filename_suffix")
BetaZero.save_solver(solver, "data/solver_$filename_suffix")
