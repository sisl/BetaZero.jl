solver = BetaZeroSolver(pomdp=pomdp,
                        updater=up,
                        belief_reward=lightdark_belief_reward,
                        params=BetaZeroParameters(
                            n_iterations=50,
                            n_data_gen=500
                        ),
                        collect_metrics=true,
                        verbose=true,
                        save_plots=true,
                        plot_incremental_data_gen=true,
                        accuracy_func=lightdark_accuracy_func)

# Neural network
solver.nn_params.training_epochs = 50 # 100
solver.nn_params.n_samples = 100_000 # 50_000
solver.nn_params.verbose_update_frequency = 100
solver.nn_params.batchsize = 1024 # 512
solver.nn_params.learning_rate = 1e-4 # 1e-3
solver.nn_params.λ_regularization = 1e-5
solver.nn_params.use_dropout = true
solver.nn_params.p_dropout = 0.2

solver.expert_results = (expert_accuracy=[0.84, 0.037], expert_returns=[11.963, 1.617], expert_label="LAVI") # LAVI baseline

# 100 epochs, 50k data, 512 batch, 1e-4 lr, and QN selection worked well? Slow learning, but continued to increase...
# 50 epochs, ___100k data___, 1024 batch, 1e-4 lr, and QN selection worked really well. Jumped right up to 0.9 accuracy.
