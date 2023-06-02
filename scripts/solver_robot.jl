solver = BetaZeroSolver(pomdp=pomdp,
                        updater=up,
                        belief_reward=robot_belief_reward,
                        params=BetaZeroParameters(
                            n_iterations=100,
                            n_data_gen=100
                        ),
                        collect_metrics=true,
                        verbose=true,
                        plot_incremental_data_gen=true)

# Neural network
solver.nn_params.n_samples = 5000
solver.nn_params.verbose_update_frequency = 100
solver.nn_params.batchsize = 64
solver.nn_params.layer_size = 256
solver.nn_params.learning_rate = 5e-5
solver.nn_params.λ_regularization = 1e-6 
solver.nn_params.verbose_plot_frequency = solver.nn_params.training_epochs
solver.nn_params.use_dropout = true
solver.nn_params.p_dropout = 0.5
solver.nn_params.use_batchnorm = true

solver.nn_params.normalize_input = true
solver.nn_params.normalize_output = true

# MCTS parameters
solver.mcts_solver.n_iterations = 2000
solver.mcts_solver.depth = 200
