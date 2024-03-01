solver = BetaZeroSolver(pomdp=pomdp,
                        updater=up,
                        belief_reward=lightdark_belief_reward,
                        params=BetaZeroParameters(
                            n_iterations=30,
                            n_data_gen=500,
                        ),
                        collect_metrics=true,
                        verbose=true,
                        save_plots=true,
                        plot_incremental_data_gen=true)

# Neural network
solver.nn_params.training_epochs = 50
solver.nn_params.n_samples = 100_000
solver.nn_params.verbose_update_frequency = 100
solver.nn_params.batchsize = 1024
solver.nn_params.learning_rate = 1e-4
solver.nn_params.λ_regularization = 1e-5
solver.nn_params.use_dropout = true
solver.nn_params.p_dropout = 0.2

solver.expert_results = (expert_accuracy=[0.84, 0.037], expert_returns=[11.963, 1.617], expert_label="LAVI") # LAVI baseline
# solver.expert_results = (expert_accuracy=[0.0, 0.0], expert_returns=[3.55, 0.15], expert_label="LAVI [LD(5)]") # LAVI baseline
