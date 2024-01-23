if is_constrained
    pomdp.incorrect_r = 0 # No explicit failure penalty
end

solver = BetaZeroSolver(pomdp=pomdp,
                        updater=up,
                        belief_reward=lightdark_belief_reward,
                        is_constrained=is_constrained,
                        params=BetaZeroParameters(
                            n_buffer=1,
                            n_iterations=50,
                            n_data_gen=500,
                        ),
                        collect_metrics=true,
                        verbose=true,
                        save_plots=true,
                        plot_incremental_data_gen=true)

# CBZ-MCTS
solver.mcts_solver.n_iterations = 500

if solver.is_constrained
    solver.mcts_solver.Δ0 = 0.01
    solver.mcts_solver.η = 0.00001
end

solver.mcts_solver.k_action = 3
solver.mcts_solver.alpha_action = 0.7
solver.mcts_solver.k_state = 5
solver.mcts_solver.alpha_state = 0.12

solver.nn_params.training_epochs = 50
solver.nn_params.n_samples = 100_000
solver.nn_params.batchsize = 1024
solver.nn_params.learning_rate = 1e-4
solver.nn_params.λ_regularization = 1e-5
solver.nn_params.use_dropout = true
solver.nn_params.p_dropout = 0.4

solver.expert_results = (expert_accuracy=[1 - 0.024, 0.0], expert_returns=[13.336207494301647, 0.0], expert_label="BetaZero(λ)")
