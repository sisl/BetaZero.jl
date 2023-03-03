global RUN_PARALLEL = true

RUN_PARALLEL && Sys.islinux() && include("launch_remote.jl")
using Revise
using Distributed

@everywhere begin
    using Revise
    using Parameters
    using ParticleFilters
    using POMDPModels
    using POMDPs
    using POMDPTools
    using Plots; default(fontfamily="Computer Modern", framestyle=:box)
    using Random
    using Statistics
    using StatsBase
    using BSON
    using BetaZero
    using LightDark
    using Flux
    include("lightdark_representation.jl")
    include("plot_lightdark.jl")

    using LocalApproximationValueIteration
    using LocalFunctionApproximation
    using GridInterpolations

    if !@isdefined(lavi_policy)
        # serialization issues when re-loading policy
        lavi_policy = BSON.load(joinpath(@__DIR__, "..", "policy_lavi.bson"))[:policy]
    end
end

pomdp = LightDark.LightDarkPOMDP()
up = LightDark.ParticleHistoryBeliefUpdater(BootstrapFilter(pomdp, 500))

filename_suffix = "lightdark_64relu.bson"

if RUN_PARALLEL
    @everywhere include("lightdark_representation.jl")

    solver = BetaZeroSolver(pomdp=pomdp,
                            updater=up,
                            params=BetaZeroParameters(
                                n_iterations=20,
                                n_data_gen=500,
                                n_evaluate=0,
                                n_holdout=0,
                                n_buffer=2
                            ),
                            belief_reward=lightdark_belief_reward,
                            collect_metrics=true,
                            verbose=true,
                            plot_incremental_data_gen=true,
                            accuracy_func=lightdark_accuracy_func)

    # Neural network
    solver.nn_params.n_samples = 1000
    solver.nn_params.verbose_update_frequency = 100
    solver.nn_params.batchsize = 256

    @show solver.params.n_buffer

    # Plotting
    solver.expert_results = (expert_accuracy=[0.84, 0.037], expert_returns=[11.963, 1.617], expert_label="LAVI") # LAVI baseline

    # initial_ensemble = BetaZero.initialize_ensemble(solver, 3)
    # policy = solve(solver, pomdp; surrogate=initial_ensemble)
    policy = solve(solver, pomdp)
    display(value_and_policy_plot(pomdp, policy))

    BetaZero.save_policy(policy, "data/policy_$filename_suffix")
    BetaZero.save_solver(solver, "data/solver_$filename_suffix")

    # display(plot_lightdark(pomdp, policy, up)) # Single episode trajectory example
else
    using BetaZero.MCTS
    using BetaZero.GaussianProcesses
    using BetaZero.DataStructures
    policy = BetaZero.load_policy(joinpath(@__DIR__, "..", "..", "data", "policy_$filename_suffix.bson"))
    solver = BetaZero.load_solver(joinpath(@__DIR__, "..", "..", "data", "solver_$filename_suffix.bson"))
end
