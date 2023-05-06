Sys.islinux() && include("launch_remote.jl")
using Revise
using Distributed

@everywhere begin
    using Revise
    using BetaZero
    include("representation_lightdark.jl")
    include("plot_lightdark.jl")
end

filename_suffix = "lightdark.bson"

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
solver.nn_params.training_epochs = 100
solver.nn_params.n_samples = 50_000
solver.nn_params.verbose_update_frequency = 100
solver.nn_params.batchsize = 512
solver.nn_params.learning_rate = 1e-3
solver.nn_params.λ_regularization = 1e-5

solver.nn_params.use_dropout = true
solver.nn_params.p_dropout = 0.2

# Plotting
solver.expert_results = (expert_accuracy=[0.84, 0.037], expert_returns=[11.963, 1.617], expert_label="LAVI") # LAVI baseline

policy = solve(solver, pomdp)
BetaZero.save_policy(policy, "data/policy_$filename_suffix")
BetaZero.save_solver(solver, "data/solver_$filename_suffix")

value_and_policy_plot(pomdp, policy)
savefig("value_policy_plot.png")
# display(plot_lightdark(pomdp, policy, up)) # Single episode trajectory example
