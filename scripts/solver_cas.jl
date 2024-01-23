if is_constrained
    pomdp.reward_collision = 0 # No explicit failure penalty
end

solver = BetaZeroSolver(pomdp=pomdp,
                        updater=up,
                        belief_reward = mean_belief_reward,
                        is_constrained=is_constrained,
                        params=BetaZeroParameters(
                            n_buffer=3,
                            n_iterations=100,
                            n_data_gen=100,
                            max_steps=pomdp.τ_max+1,
                        ),
                        collect_metrics=true,
                        verbose=true,
                        save_plots=true,
                        plot_incremental_data_gen=true)

if solver.is_constrained
    solver.mcts_solver.Δ0 = 0.01
    solver.mcts_solver.η = 0.1
    solver.mcts_solver.δ = 0.975 # (1 - 1/τ_max = 0.975)
end

# Neural network
solver.nn_params.training_epochs = 50
solver.nn_params.n_samples = 100_000
solver.nn_params.batchsize = 1024
solver.nn_params.learning_rate = 1e-4
solver.nn_params.λ_regularization = 1e-5
solver.nn_params.use_dropout = false

# MCTS parameters
solver.mcts_solver.n_iterations = 500
solver.mcts_solver.k_action = 3
solver.mcts_solver.alpha_action = 0.7
solver.mcts_solver.k_state = 5
solver.mcts_solver.alpha_state = 0.12

solver.mcts_solver.exploration_constant = 10
if is_constrained
    solver.mcts_solver.final_criterion = MCTS.SampleZQNS(τ=1)
else
    solver.mcts_solver.final_criterion = MCTS.SampleZQN(τ=1)
end
