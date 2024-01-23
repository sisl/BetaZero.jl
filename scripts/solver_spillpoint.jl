if is_constrained
    # Handled in `representation_spillpoint.jl` on immutable POMDP
    # pomdp.exited_reward_amount = 0 # No explicit failure penalty
    # pomdp.exited_reward_binary = 0 # No explicit failure penalty
end

solver = BetaZeroSolver(pomdp=pomdp,
                        updater=up,
                        belief_reward = mean_belief_reward′,
                        is_constrained=is_constrained,
                        params=BetaZeroParameters(
                            # n_iterations=2, # Used to precompile Julia faster.
                            # n_data_gen=1,
                            # max_steps=1,
                            n_buffer=3,
                            n_iterations=25,
                            n_data_gen=128,
                            max_steps=25,
                        ),
                        collect_metrics=true,
                        verbose=true,
                        save_plots=true,
                        plot_incremental_data_gen=true)

if solver.is_constrained
    solver.mcts_solver.Δ0 = 0.05
    solver.mcts_solver.η = 0.5
    solver.mcts_solver.δ = 0.5
end

if is_constrained
    solver.mcts_solver.final_criterion = MCTS.SampleZQNS(τ=0.5) # argmax as τ → 0
else
    solver.mcts_solver.final_criterion = MCTS.SampleZQN(τ=0.5) # argmax as τ → 0
end

# Neural network
solver.nn_params.training_epochs = 50
solver.nn_params.n_samples = 100_000
solver.nn_params.batchsize = 1024
solver.nn_params.layer_size = 128
solver.nn_params.network_depth = 4
solver.nn_params.loss_func = BetaZero.Flux.mae
solver.nn_params.learning_rate = 5e-6
solver.nn_params.λ_regularization = 1e-5
solver.nn_params.normalize_input = true
solver.nn_params.zero_out_tried_actions = true

solver.expert_results = (expert_accuracy=[1.0, 0.0], expert_returns=[2.24, 4.04/sqrt(10)], expert_label="POMCPOW (SIR)")

# MCTS parameters
solver.mcts_solver.n_iterations = 50
solver.mcts_solver.exploration_constant = 20.0
solver.mcts_solver.k_state = 10.0
solver.mcts_solver.alpha_state = 0.3
solver.mcts_solver.k_action = 10.0
solver.mcts_solver.alpha_action = 0.5
solver.mcts_solver.depth = 5
