if is_constrained
    pomdp.incorrect_r = 0 # No explicit failure penalty
end

solver = BetaZeroSolver(pomdp=pomdp,
                        updater=up,
                        belief_reward=lightdark_belief_reward,
                        is_constrained=is_constrained,
                        params=BetaZeroParameters(
                            n_buffer=1,
                            n_iterations=is_constrained ? 50 : 30,
                            n_data_gen=100,
                        ),
                        collect_metrics=true,
                        verbose=true,
                        save_plots=true,
                        plot_incremental_data_gen=true)
if use_despot
    random = solve(RandomSolver(), pomdp)
    bds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(random), pomdp.correct_r, check_terminal=true)
    # solver.mcts_solver = DESPOTSolver(; default_action=random, lambda=10.0, K=30, T_max=1, D=90, bounds=bds, bounds_warnings=false)

    # solver.mcts_solver = DESPOTSolver(; default_action=random, lambda=0.1, K=100, T_max=1, D=90, bounds=bds, bounds_warnings=false) # LightDark(5)

    # solver.mcts_solver = DESPOTSolver(; default_action=random, lambda=0.1, K=30, T_max=1, D=90, bounds=bds, bounds_warnings=false) # NOTE: Same as `baselines.jl`
    
    solver.mcts_solver = DESPOTSolver(; default_action=random, lambda=100.0, K=500, T_max=1, D=50, bounds=bds, bounds_warnings=false)

    # solver.mcts_solver = DESPOTSolver(; default_action=random, lambda=100.0, K=500, T_max=0.1, D=50, bounds=bds, bounds_warnings=false) # Old LD(10)
    # solver.mcts_solver = DESPOTSolver(; default_action=random, lambda=0.1, K=500, T_max=1, D=50, bounds=bds, bounds_warnings=false)
else
    # CBZ-MCTS
    solver.mcts_solver.n_iterations = is_constrained ? 500 : 100

    if solver.is_constrained
        solver.mcts_solver.Δ0 = 0.01
        solver.mcts_solver.η = 0.00001

        solver.mcts_solver.k_action = 3
        solver.mcts_solver.alpha_action = 0.7
        solver.mcts_solver.k_state = 5
        solver.mcts_solver.alpha_state = 0.12
    end
end

solver.nn_params.training_epochs = 50
solver.nn_params.n_samples = 100_000
solver.nn_params.batchsize = 1024
solver.nn_params.learning_rate = 1e-4
solver.nn_params.λ_regularization = 1e-5
solver.nn_params.use_dropout = true
solver.nn_params.p_dropout = is_constrained ? 0.4 : 0.2

if is_constrained
    solver.expert_results = (expert_accuracy=[1 - 0.024, 0.0], expert_returns=[13.336207494301647, 0.0], expert_label="BetaZero(λ)")
else
    solver.expert_results = (expert_accuracy=[0.84, 0.037], expert_returns=[11.963, 1.617], expert_label="LAVI") # LAVI baseline
end
