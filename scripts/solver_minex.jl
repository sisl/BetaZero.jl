solver = BetaZeroSolver(pomdp=pomdp,
                        updater=up,
                        belief_reward=simple_minex_belief_reward,
                        params=BetaZeroParameters(
                            n_iterations=20,
                            n_data_gen=100,
                            n_holdout=50
                        ),
                        collect_metrics=true,
                        verbose=true,
                        save_plots=true,
                        plot_incremental_data_gen=true,
                        plot_incremental_holdout=true,
                        plot_metrics_filename="intermediate_metrics_figure_minex.png")

# MCTS parameters
solver.mcts_solver = PUCTSolver(n_iterations=50,
    exploration_constant=57.0,
    k_action=41.09,
    alpha_action=0.5657,
    k_state=37.13,
    alpha_state=0.9394,
    depth=5,
    final_criterion=MCTS.SampleZQN(τ=1, zq=1, zn=1),
    counts_in_info=true)

# Neural network parameters
solver.nn_params.use_cnn = true
solver.nn_params.training_epochs = 10
solver.nn_params.n_samples = 100_000
solver.nn_params.learning_rate = 1e-6
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

solver.expert_results = (expert_accuracy=[0.84, 0.04], expert_returns=[-5.04, 0.861], expert_label="POMCPOW")
