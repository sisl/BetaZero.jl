Sys.islinux() && include("launch_remote.jl")
using Revise
using Distributed

@everywhere begin
    using Revise
    using BetaZero
    include("representation_rocksample.jl")
end

filename_suffix = "rocksample"

solver = BetaZeroSolver(pomdp=pomdp,
                        updater=up,
                        belief_reward=rocksample_belief_reward,
                        params=BetaZeroParameters(
                            n_iterations=50,
                            n_data_gen=500
                        ),
                        collect_metrics=true,
                        verbose=true,
                        plot_incremental_data_gen=true,
                        accuracy_func=rocksample_accuracy_func)

# Neural network
solver.nn_params.training_epochs = 100 # WAS: 1000
solver.nn_params.n_samples = 100_000 # WAS: 5000 # WAS: 1000
solver.nn_params.verbose_update_frequency = 100
solver.nn_params.batchsize = 1024 # WAS: 256
solver.nn_params.learning_rate = 1e-3
solver.nn_params.λ_regularization = 1e-5

solver.nn_params.use_dirichlet_exploration = false # NOTE.
# solver.nn_params.α_dirichlet = 1 # Default 0.03
# solver.nn_params.ϵ_dirichlet = 0.25 # Default 0.25

solver.nn_params.use_dropout = true
solver.nn_params.p_dropout = 0.2

# MCTS parameters
solver.mcts_solver = PUCTSolver(n_iterations=200,
    exploration_constant=1.0,
    k_action=10.0, # NOTE: was 2
    alpha_action=0.5, # WAS: 0.25
    k_state=2.0,
    alpha_state=0.1,
    counts_in_info=true)

policy = solve(solver, pomdp)
BetaZero.save_policy(policy, "data/policy_$filename_suffix")
BetaZero.save_solver(solver, "data/solver_$filename_suffix")
