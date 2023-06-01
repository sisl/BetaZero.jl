solver = BetaZeroSolver(pomdp=pomdp,
                        updater=up,
                        belief_reward=rocksample_belief_reward,
                        params=BetaZeroParameters(
                            n_iterations=50,
                            n_data_gen=500,
                            n_holdout=100,
                        ),
                        collect_metrics=true,
                        verbose=true,
                        save_plots=true,
                        plot_incremental_data_gen=true,
                        plot_incremental_holdout=true,
                        accuracy_func=rocksample_accuracy_func)

# Neural network
solver.nn_params.training_epochs = 10
solver.nn_params.n_samples = 100_000
solver.nn_params.verbose_update_frequency = 100
solver.nn_params.batchsize = 1024
solver.nn_params.learning_rate = 1e-3
solver.nn_params.λ_regularization = 1e-5
solver.nn_params.use_dropout = true
solver.nn_params.p_dropout = 0.5
solver.nn_params.optimizer = BetaZero.Flux.RMSProp
solver.nn_params.use_batchnorm = true
solver.nn_params.batchnorm_momentum = 0.7f0
solver.nn_params.layer_size = 128
solver.nn_params.incremental_save = true
solver.nn_params.policy_filename = "betazero_policy_rs1515.bson"

solver.expert_results = (expert_accuracy=[0.0, 0.0], expert_returns=[17.16, 0.21], expert_label="AdaOPS")

# MCTS parameters
solver.mcts_solver = PUCTSolver(
    n_iterations=100,
    exploration_constant=50.0,
    enable_action_pw=false,
    enable_state_pw=false,
    depth=15,
    final_criterion=MCTS.SampleZQN(τ=1, zq=1, zn=1))
