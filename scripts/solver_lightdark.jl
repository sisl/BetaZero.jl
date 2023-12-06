is_cc = true
solver = BetaZeroSolver(pomdp=pomdp,
                        updater=up,
                        belief_reward=lightdark_belief_reward,
                        is_cc=is_cc, # ! NOTE. TODO: Rename to is_constrained
                        params=BetaZeroParameters(
                            n_buffer=is_cc ? 3 : 1, # 3 # ! NOTE.
                            n_iterations=50, # 200
                            n_data_gen=is_cc ? 500 : 500, # 127 # 500, # 50, # 500, # 100,
                            # n_holdout=500,
                        ),
                        collect_metrics=true,
                        verbose=true,
                        save_plots=true,
                        # plot_incremental_data_gen=true,
                        # plot_incremental_holdout=true)
                        plot_incremental_data_gen=true)

# CBZ-MCTS
# solver.mcts_solver.final_criterion = MaxZQN() # ! NOTE: BetaZero testing.
# solver.mcts_solver.final_criterion = SampleZQNS(τ=0.5) # 0.5 # ! NOTE: Explore.

if solver.is_cc
    solver.mcts_solver.n_iterations = 500 # 500 # ! NOTE.
    solver.mcts_solver.depth = 50 # 50 # ! NOTE.
    # solver.mcts_solver.exploration_constant = 10.0 # 1.0 # 2.0 # 10.0

    solver.mcts_solver.α0 = 0.01 # 0.01 # 0.001 (w/η=0.0001) # 0.005 (w/η=0.0005) 0.05 (w/η=0.01) # 0.01 (w/η=0.005)
    solver.mcts_solver.η = 0.0005 # 0.005 # 0.05 # 0.0025 # 0.005 # 0.01 # 0.005 # 0.1 # 0.02 # 0.1/5 # 0.1
    # solver.mcts_solver.η = 0.005 # 0.05 # 0.0025 # 0.005 # 0.01 # 0.005 # 0.1 # 0.02 # 0.1/5 # 0.1
    # solver.mcts_solver.η = 0.001 # 0.05 # 0.0025 # 0.005 # 0.01 # 0.005 # 0.1 # 0.02 # 0.1/5 # 0.1
    # solver.mcts_solver.η = 0.05 # 0.05 # 0.0025 # 0.005 # 0.01 # 0.005 # 0.1 # 0.02 # 0.1/5 # 0.1

    # solver.mcts_solver.enable_action_pw = false # ! NOTE.
    # solver.mcts_solver.k_action = 5.0 # 5.0 # 12.0 # 2.0
    # solver.mcts_solver.alpha_action = 0.5 # 0.5 # 0.1
    # solver.nn_params.zero_out_tried_actions = true # ! NOTE: Expand on other actions
    
    # solver.mcts_solver.k_state = 12.0 # 5.0 # 12.0 # 6.0 # 2.0 # ! NOTE.
    # solver.mcts_solver.alpha_state = 0.5 # 0.1 # 0.5 # 0.1 # 0.1 # ! NOTE.
end

# solver.mcts_solver.k_action = 3.0 # 2.0
# solver.mcts_solver.alpha_action = 0.0 # 0.1
# solver.mcts_solver.n_iterations = 200 # 100
# solver.mcts_solver.exploration_constant = 10.0 # 1 # ! NOTE.

# CPUCT-MCTS
# solver.mcts_solver = CPUCTSolver(solver.mcts_solver)
# solver.mcts_solver.final_criterion = MaxZQNS(zq=1, zn=1)
# solver.mcts_solver.final_criterion = SampleZQNS(zq=1, zn=1)

# Neural network
solver.nn_params.training_epochs = 50 # 50 # 50 # 10 # 50 # 20 # 10 # 50
solver.nn_params.n_samples = 100_000 # 10_000 # 5_000 # 10_000 # 100_000 # 5_000 # 2_000 # 100_000
solver.nn_params.verbose_update_frequency = 100
solver.nn_params.batchsize = 1024 # 256 # 120 # 64 # 1024 # 128 # 1024
solver.nn_params.learning_rate = 1e-4 # 1e-4 # 1e-3 # 1e-4
solver.nn_params.λ_regularization = 1e-5
solver.nn_params.use_dropout = true
solver.nn_params.p_dropout = 0.2

# Simulation parameters
# solver.params.max_steps = 500 # 100 # Important for safety case, takes longer to collect information.

# solver.expert_results = (expert_accuracy=[0.84, 0.037], expert_returns=[11.963, 1.617], expert_label="LAVI") # LAVI baseline
# solver.expert_results = (expert_accuracy=[0.0, 0.0], expert_returns=[3.55, 0.15], expert_label="LAVI [LD(5)]") # LAVI baseline
