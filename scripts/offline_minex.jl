Sys.islinux() && include("launch_remote.jl")
using Revise
using Distributed

@everywhere begin
    using Revise
    using BetaZero
    include("representation_minex.jl")
end

@everywhere filename_suffix = "mineral_exploration.bson"

solver = BetaZeroSolver(pomdp=pomdp,
                        updater=up,
                        belief_reward=simple_minex_belief_reward,
                        params=BetaZeroParameters(
                            n_iterations=50,
                            n_data_gen=100
                        ),
                        collect_metrics=true,
                        verbose=true,
                        save_plots=true,
                        plot_incremental_data_gen=true,
                        accuracy_func=simple_minex_accuracy_func)

# MCTS parameters
solver.mcts_solver = PUCTSolver(n_iterations=50,
    exploration_constant=57.0,
    k_action=41.09,
    alpha_action=0.5657,
    k_state=37.13,
    alpha_state=0.9394,
    depth=5,
    counts_in_info=true)

# Neural network parameters
solver.nn_params.use_cnn = true
solver.nn_params.training_epochs = 10 # 100
solver.nn_params.n_samples = 20_000
solver.nn_params.learning_rate = 1e-5 # 1e-6
solver.nn_params.λ_regularization = 1e-4
solver.nn_params.batchsize = 1024
solver.nn_params.layer_size = 256
solver.nn_params.optimizer = BetaZero.Flux.RMSProp
solver.nn_params.loss_func = BetaZero.Flux.mae
solver.nn_params.zero_out_tried_actions = true

solver.nn_params.normalize_input = false
solver.nn_params.use_dropout = true
solver.nn_params.p_dropout = 0.7
solver.nn_params.use_batchnorm = true
solver.nn_params.batchnorm_momentum = 0.7f0
solver.nn_params.incremental_save = true

solver.expert_results = (expert_accuracy=[0.574, 0.011], expert_returns=[-8.9, 0.338], expert_label="POMCPOW") # POMCPOW baseline (relative returns, no heuristic estimate value, across 2000 episodes)

policy = solve(solver, pomdp)
BetaZero.save_policy(policy, "policy_$filename_suffix")
BetaZero.save_solver(solver, "solver_$filename_suffix")
