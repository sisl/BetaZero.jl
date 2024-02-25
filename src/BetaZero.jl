module BetaZero

using Reexport

include(joinpath(@__DIR__, "..", "submodules", "MCTS", "src", "MCTS.jl"))
@reexport using .MCTS

@reexport using BSON
@reexport using DataStructures
@reexport using Flux
@reexport using Flux.NNlib
@reexport using Flux.MLUtils
@reexport using GaussianProcesses
@reexport using Plots; default(fontfamily="Computer Modern", framestyle=:box)
@reexport using ParticleFilters
@reexport using POMDPs
@reexport using Random
@reexport using StatsBase
using Distributions
using Distributed
using Metalhead
using JLD2
using LinearAlgebra
using Optim
using Parameters
using Pkg
using POMDPTools
using ProgressMeter
using Statistics
using Suppressor
using UnicodePlots
import Flux.Zygote: ignore_derivatives


include("belief_mdp.jl")
include("interface.jl")
include("parameters.jl")
include("running_stats.jl")

export
    BetaZeroSolver,
    BetaZeroPolicy,
    BetaZeroParameters,
    BetaZeroNetworkParameters,
    BetaZeroGPParameters,
    BeliefMDP,
    RawNetworkPolicy,
    RawValueNetworkPolicy,
    initialize_network,
    calc_loss_weight,
    value_plot,
    policy_plot,
    value_and_policy_plot,
    value_policy_uncertainty_plot,
    uncertainty_plot,
    plot_accuracy,
    plot_returns,
    plot_accuracy_and_returns,
    plot_online_performance,
    plot_data_gen,
    plot_ablations,
    bettersavefig,
    save_policy,
    save_solver,
    save_surrogate,
    load_policy,
    load_solver,
    load_surrogate,
    load_incremental,
    mean_and_stderr,
    solve_planner!,
    network_input,
    network_lookup,
    value_lookup,
    policy_lookup,
    dirichlet_noise,
    bootstrap,
    install_extras,
    accuracy,
    mean_belief_reward,
    mean_belief_reward′

mean_belief_reward(pomdp::POMDP, b, a, bp) = mean(reward(pomdp, s, a) for s in particles(b)) # reward function: R(b,a,b′), uses R(s,a)
mean_belief_reward′(pomdp::POMDP, b, a, bp) = mean(reward(pomdp, s, a, sp) for (s,sp) in zip(particles(b), particles(bp))) # reward function: R(b,a,b′), uses R(s,a,sp)

@with_kw mutable struct BetaZeroSolver <: POMDPs.Solver
    pomdp::POMDP
    updater::POMDPs.Updater
    params::BetaZeroParameters = BetaZeroParameters() # parameters for BetaZero algorithm
    nn_params::BetaZeroNetworkParameters = BetaZeroNetworkParameters(input_size=get_input_size(pomdp,updater), action_size=length(actions(pomdp))) # parameters for training NN
    gp_params::BetaZeroGPParameters = BetaZeroGPParameters(input_size=get_input_size(pomdp,updater)) # parameters for training GP
    data_buffer_train::CircularBuffer = CircularBuffer(params.n_buffer) # Simulation data buffer for training (Note: Each simulation has multiple time steps of data)
    data_buffer_valid::CircularBuffer = CircularBuffer(params.n_buffer) # Simulation data buffer for validation (Note: Making sure to clearly separate training from validation to prevent data leakage)
    bmdp::Union{BeliefMDP,Nothing} = nothing # Belief-MDP version of the POMDP
    belief_reward::Function = mean_belief_reward
    include_info::Bool = false # Include `action_info` in metrics when running POMDP simulation
    mcts_solver::AbstractMCTSSolver = PUCTSolver(n_iterations=100,
                                                check_repeat_action=true,
                                                exploration_constant=1.0,
                                                k_action=2.0,
                                                alpha_action=0.25,
                                                k_state=2.0,
                                                alpha_state=0.1,
                                                tree_in_info=false,
                                                counts_in_info=true, # Note, required for policy vector.
                                                show_progress=false,
                                                final_criterion=MaxZQN(zq=1, zn=1),
                                                estimate_value=(bmdp,b,d)->0.0) # `estimate_value` will be replaced with a surrogate lookup
    data_collection_policy::Policy = RandomPolicy(Random.GLOBAL_RNG, pomdp, updater) # Policy used for data collection (if indicated to use different policy than the BetaZero on-policy)
    use_data_collection_policy::Bool = false # Use provided policy for data collection.
    collect_metrics::Bool = true # Indicate that performance metrics should be collected.
    performance_metrics::Array = [] # Stored metrics for data generation runs.
    holdout_metrics::Array = [] # Metrics computed from holdout test set.
    plot_incremental_data_gen::Bool = false # Plot accuracies and returns over iterations after data collection
    plot_incremental_holdout::Bool = false # Plot accuracies and returns over iterations after running holdout test
    display_plots::Bool = plot_incremental_data_gen || plot_incremental_holdout # Display metrics plots after data collection
    save_plots::Bool = false # Save metrics plots after data collection
    plot_metrics_filename::String = "intermediate_metrics_figure.png"
    expert_results::NamedTuple{(:expert_accuracy, :expert_returns, :expert_label), Tuple{Union{Nothing,Vector}, Union{Nothing,Vector}, String}} = (expert_accuracy=nothing, expert_returns=nothing, expert_label="expert") # For plotting comparisons, pass in the [mean, stderr] for the expert metrics.
    verbose::Bool = true # Print out debugging/training/simulation information during solving
end


@with_kw mutable struct BetaZeroTrainingData
    b = nothing # current belief
    π = nothing # current policy estimate (using N(s,a))
    z = nothing # final discounted return of the episode
end


# Needs BetaZeroSolver defined.
include("utils.jl")
include("metrics.jl")
include("gaussian_process.jl")
include("ensemble.jl")


const Surrogate = Union{Chain, GPSurrogate, EnsembleNetwork} # Needs GPSurrogate and EnsembleNetwork defined.

mutable struct BetaZeroPolicy <: POMDPs.Policy
    surrogate::Surrogate
    planner::AbstractMCTSPlanner
    parameters::ParameterCollection
end


include("neural_network.jl") # Needs BetaZeroPolicy and RunningStats
include("raw_network.jl") # Needs Surrogate defined.
include("saving.jl") # Needs BetaZeroSolver and BetaZeroPolicy defined.
include("plots.jl")


"""
The main BetaZero policy iteration algorithm.
"""
function POMDPs.solve(solver::BetaZeroSolver, pomdp::POMDP; surrogate::Surrogate=solver.params.use_nn ? initialize_network(solver) : initialize_gaussian_process(solver), resume::Bool=false)
    if (solver.params.n_ensembled > 1)
        surrogate = initialize_ensemble(solver, solver.params.n_ensembled)
    end
    local current_surrogate = surrogate
    local best_surrogate = surrogate
    rstats = RunningStats()
    check_data_buffer_size!(solver)
    fill_bmdp!(solver)

    N = ceil(Int, length(solver.performance_metrics) / solver.params.n_data_gen)

    @conditional_time solver.verbose for i in 1:solver.params.n_iterations
        i = resume ? N + i : i
        solver.verbose && println(); println("—"^40); println(); @info "BetaZero iteration $i/$(solver.params.n_iterations + N)"

        if i > 1
            # x) Evaluate BetaZero agent (compare to previous agent).
            best_surrogate = evaluate_agent(pomdp, solver, best_surrogate, current_surrogate; outer_iter=typemax(Int32)+i)
        end

        # 0) Evaluate performance on a holdout test set (never used for training or surrogate selection).
        run_holdout_test!(pomdp, solver, best_surrogate)

        # x) Generate data using the best BetaZero agent so far: {[belief, return], ...}
        use_different_policy = (i == 1) ? solver.use_data_collection_policy : false # only use data collection policy on first iteration
        data, metrics = generate_data(pomdp, solver, best_surrogate; use_different_policy=use_different_policy, inner_iter=solver.params.n_data_gen, outer_iter=i)

        # x) Store generated data from the best surrogate.
        store_data!(solver, data)
        store_metrics!(solver, metrics)

        # x) Optimize surrogate with recent simulated data.
        if i != solver.params.n_iterations
            current_surrogate = train(deepcopy(best_surrogate), solver; rstats, verbose=solver.verbose)
        end

        # Save off incremental policy
        incremental_save(solver, best_surrogate, i)
    end

    # Include the surrogate in the MCTS planner as part of the BetaZero policy
    return solve_planner!(solver, best_surrogate)
end


"""
Conver the `POMDP` to a `BeliefMDP` and set the `pomdp.bmdp` field.
"""
function fill_bmdp!(solver::BetaZeroSolver)
    solver.bmdp = BeliefMDP(solver.pomdp, solver.updater, solver.belief_reward)
    return solver.bmdp
end


"""
Return the BetaZero planner, first adding the value estimator and then solving the inner MCTS planner.
"""
function solve_planner!(policy::BetaZeroPolicy, f::Surrogate=policy.surrogate)
    attach_surrogate!(policy.planner.solver, policy.parameters.nn_params, f)
    bmdp = policy.planner.mdp
    policy.planner.solver.reset_callback = (mdp, s)->false
    policy.planner.solver.timer = ()->1e-9 * time_ns()
    if policy.planner.solver.init_Q isa Function || policy.parameters.params.bootstrap_q
        policy.planner.solver.init_Q = bootstrap(f) # re-apply bootstrap
    end
    if policy.parameters.params.bootstrap_u && f isa EnsembleNetwork
        policy.planner.solver.init_U = bootstrap_uncertainty(f) # apply ensemble bootstrap
    end
    mcts_planner = solve(policy.planner.solver, bmdp)
    policy.surrogate = f
    policy.planner = mcts_planner
    return policy
end

function solve_planner!(solver::BetaZeroSolver, f::Surrogate)
    attach_surrogate!(solver, f)
    fill_bmdp!(solver)
    solver.mcts_solver.reset_callback = (mdp, s)->false
    solver.mcts_solver.timer = ()->1e-9 * time_ns()
    mcts_planner = solve(solver.mcts_solver, solver.bmdp)
    parameters = ParameterCollection(solver.params, solver.nn_params, solver.gp_params)
    return BetaZeroPolicy(f, mcts_planner, parameters)
end


"""
Attach the surrogate model to the MCTS solver for value estimates and next action selection.
"""
attach_surrogate!(solver::BetaZeroSolver, f::Surrogate) = attach_surrogate!(solver.mcts_solver, solver.nn_params, f)
function attach_surrogate!(mcts_solver, nn_params::BetaZeroNetworkParameters, f::Surrogate)
    if mcts_solver isa GumbelSolver
        mcts_solver.estimate_value=(bmdp,b,d)->value_lookup(f, b)
        mcts_solver.estimate_policy=(bmdp,b)->policy_lookup(f, b)
    elseif mcts_solver isa DARSolver
        mcts_solver.estimate_value=(bmdp,b,d)->value_lookup(f, b)
        mcts_solver.estimate_policy=(bmdp,b)->policy_lookup(f, b)
        mcts_solver.next_action = (bmdp,b,bnode)->next_action(bmdp, b, f, nn_params, bnode)
    elseif mcts_solver isa PUCTSolver
        mcts_solver.estimate_value=(bmdp,b,d)->value_lookup(f, b)
        mcts_solver.estimate_policy=(bmdp,b)->policy_lookup(f, b)
        mcts_solver.next_action = (bmdp,b,bnode)->next_action(bmdp, b, f, nn_params, bnode)
    else
        mcts_solver.estimate_value = (bmdp,b,d)->value_lookup(f, b)
        mcts_solver.next_action = (bmdp,b,bnode)->next_action(bmdp, b, f, nn_params, bnode)
    end
    return nothing
end


"""
Uniformly sample data from buffer (with replacement).
Note that the buffer is per-simulation with each simulation having multiple time steps.
We want to sample `n` individual time steps across the simulations.
"""
function sample_data(data_buffer::CircularBuffer, n::Int; sample_more_than_collected::Bool=true)
    sim_times = map(d->size(d.Y,2), data_buffer) # number of time steps in each simulation
    data_buffer_indices = 1:length(data_buffer)
    belief_size = size(data_buffer[1].X)[1:end-1]
    belief_size_span = map(d->1:d, belief_size) # e.g., (1:30, 1:30, 1:5)
    num_data_points = sum(sim_times)

    if !sample_more_than_collected && n > num_data_points
        @warn "Requested more data ($n) than is available ($num_data_points). Only sampling $num_data_points data."
        n = num_data_points
    end

    sampled_sims_indices = sample(data_buffer_indices, Weights(sim_times), n; replace=true) # weighted based on num. steps per sim (to keep with __overall__ uniform across time steps)

    X = Array{Float32}(undef, belief_size..., n)
    output_size = size(data_buffer[1].Y, 1)
    Y = Array{Float32}(undef, output_size, n)
    for (i,sim_i) in enumerate(sampled_sims_indices)
        sim = data_buffer[sim_i]
        T = size(sim.Y, 2)
        t = rand(1:T) # uniformly sample time from this simulation
        setindex!(X, getindex(sim.X, belief_size_span..., t), belief_size_span..., i) # general for any size matrix e.g., X[:,;,:,i] = sim.X[:,:,:,t]
        Y[:, i] = sim.Y[:, t]
    end
    return (X=X, Y=Y)
end


"""
Compare previous and current neural networks using MCTS simulations.
Use upper confidence bound on the discounted return as the comparison metric.
"""
function evaluate_agent(pomdp::POMDP, solver::BetaZeroSolver, best_surrogate::Surrogate, current_surrogate::Surrogate; outer_iter=typemax(Int32))
    if solver.params.n_evaluate == 0
        return current_surrogate
    else
        eval_on_accuracy = solver.params.eval_on_accuracy

        function generate_evaluation_data(surrogate::Surrogate)
            data, metrics = generate_data(pomdp, solver, surrogate; inner_iter=solver.params.n_evaluate, outer_iter=outer_iter)
            return eval_on_accuracy ? [m.accuracy for m in metrics] : data.G
        end

        solver.verbose && @info "Evaluting best-so-far network..."
        criteria_prev = generate_evaluation_data(best_surrogate)

        solver.verbose && @info "Evaluting current network..."
        criteria_curr = generate_evaluation_data(current_surrogate)

        λ = solver.params.λ_ucb
        μ_prev, σ_prev = mean_and_std(criteria_prev)
        μ_curr, σ_curr = mean_and_std(criteria_curr)
        ucb_prev = μ_prev + λ*σ_prev
        ucb_curr = μ_curr + λ*σ_curr

        if ucb_curr ≥ ucb_prev
            solver.verbose && ucb_curr == ucb_prev && @info "[IDENTICAL UCBs]"
            solver.verbose && @info "<<<< New surrogate performed better [new = $ucb_curr, old = $ucb_prev] >>>>"
            return current_surrogate
        else
            solver.verbose && @info "---- Previous surrogate performed better [new = $ucb_curr, old = $ucb_prev] ----"
            return best_surrogate
        end
    end
end


"""
Store performance metrics. Duplicate stored metrics if network is not better than the previous best-so-far.
"""
function store_metrics!(solver::BetaZeroSolver, metrics)
    if solver.collect_metrics
        push!(solver.performance_metrics, metrics...)
    end

    # Plot incremental learning
    if solver.plot_incremental_data_gen
        performance_plot = plot_accuracy_and_returns(solver; include_holdout=solver.plot_incremental_holdout)
        solver.save_plots && Plots.savefig(solver.plot_metrics_filename)
        solver.display_plots && display(performance_plot)
    end
end


"""
Generate training data using online MCTS with the best surrogate so far `f` (parallelized across episodes).
"""
function generate_data(pomdp::POMDP, solver::BetaZeroSolver, f::Surrogate;
                        outer_iter::Int=0, inner_iter::Int=solver.params.n_data_gen,
                        return_metrics::Bool=true,
                        use_different_policy::Bool=false,
                        holdout_criteria::Bool=false)
    # Confirm that surrogate is on the CPU for inference
    f = cpu(f)
    up = solver.updater
    fill_bmdp!(solver)
    bmdp = solver.bmdp

    if use_different_policy
        @info "Using provided policy for data generation..."
        planner = solver.data_collection_policy
    else
        if solver.params.use_raw_policy_network
            @info "Using raw [policy] network for data generation..."
            planner = RawNetworkPolicy(pomdp, f)
        elseif solver.params.use_raw_value_network
            @info "Using raw [value] network for data generation..."
            planner = RawValueNetworkPolicy(bmdp, f)
            planner.n_obs = solver.params.raw_value_network_n_obs
            @info "Number of onestep value observations = $(planner.n_obs)"
        else
            # Run MCTS to generate data using the surrogate `f`
            attach_surrogate!(solver, f)

            if holdout_criteria
                if hasproperty(solver.mcts_solver, :final_criterion) && hasproperty(solver.mcts_solver.final_criterion, :τ)
                    @info "Changing holdout final criteria to return maximizing action."
                    mcts_solver_holdout = deepcopy(solver.mcts_solver)
                    crit = mcts_solver_holdout.final_criterion
                    mcts_solver_holdout.final_criterion = typeof(crit)(NamedTuple(p=>p == :τ ? 0 : getproperty(crit, p) for p in propertynames(crit))...) # return maximizing action (bypass immutable structs)
                    mcts_solver = mcts_solver_holdout
                else
                    @info "Running holdout with for solver $(typeof(solver.mcts_solver))"
                    mcts_solver = solver.mcts_solver
                end
            else
                mcts_solver = solver.mcts_solver
            end

            planner = solve(mcts_solver, bmdp)
        end
    end

    collect_metrics = solver.collect_metrics
    include_info = solver.include_info
    max_steps = solver.params.max_steps
    final_criterion = mcts_solver.final_criterion
    use_completed_policy_gumbel = solver.params.use_completed_policy_gumbel
    skip_missing_reward_signal = solver.params.skip_missing_reward_signal
    train_missing_on_predicted = solver.params.train_missing_on_predicted
    nn_params = solver.nn_params

    solver.verbose && @info "Number of processes: $(nprocs())"
    progress = Progress(inner_iter)
    channel = RemoteChannel(()->Channel{Bool}(), 1)

    @async while take!(channel)
        next!(progress)
    end

    @time parallel_data = pmap(i->begin
        seed = parse(Int, string(outer_iter, lpad(i, length(digits(inner_iter)), '0'))) # 1001, 1002, etc. for BetaZero outer_iter=1
        Random.seed!(seed)
        # @info "Generating data ($i/$(inner_iter)) with seed ($seed)"
        ds0 = initialstate(pomdp)
        s0 = rand(ds0)
        b0 = initialize_belief(up, ds0)
        data, metrics = run_simulation(pomdp, planner, up, b0, s0; collect_metrics, include_info, max_steps, skip_missing_reward_signal, train_missing_on_predicted, final_criterion, use_completed_policy_gumbel, nn_params)
        if ismissing(data) && ismissing(metrics)
            # ignore missing data
            B = Z = Π = metrics = discounted_return = missing
        else
            B = []
            Z = []
            Π = []
            discounted_return = data[1].z
            for d in data
                push!(B, d.b)
                push!(Z, d.z)
                push!(Π, d.π)
            end
        end
        put!(channel, true) # trigger progress bar update
        B, Z, Π, metrics, discounted_return
    end, 1:inner_iter)

    put!(channel, false) # tell printing task to finish

    beliefs = vcat([d[1] for d in parallel_data if !ismissing(d[1])]...) # combine all beliefs
    returns = vcat([d[2] for d in parallel_data if !ismissing(d[2])]...) # combine all returns
    policy_vecs = vcat([d[3] for d in parallel_data if !ismissing(d[3])]...) # combine all policy vectors
    metrics = vcat([d[4] for d in parallel_data if !ismissing(d[4])]...) # combine all metrics
    G = vcat([d[5] for d in parallel_data if !ismissing(d[5])]...) # combine all final returns

    solver.verbose && @info "Percent non-missing: $(length(G)/inner_iter*100)%"

    if solver.verbose
        μ, σ = mean_and_std(G)
        n_returns = length(G)
        accuracies = [m.accuracy for m in metrics]
        μ_acc, σ_acc = mean_and_std(accuracies)
        n_accs = length(accuracies)
        @info "Generated data return statistics: $(round(μ, digits=3)) ± $(round(σ/sqrt(n_returns), digits=3)) returns, $(round(μ_acc, digits=3)) ± $(round(σ_acc/sqrt(n_accs), digits=3)) accuracy"
    end

    # Much faster than `cat(belief...; dims=4)`
    belief = beliefs[1]
    X = Array{Float32}(undef, size(belief)..., length(beliefs))
    for i in eachindex(beliefs)
        # Generalize for any size matrix (equivalent to X[:,:,:,i] = beliefs[i] for 3D matrix)
        setindex!(X, beliefs[i], map(d->1:d, size(belief))..., i)
    end

    policy_vec = policy_vecs[1]
    output_size = 1 + length(policy_vec) # [value, policy_vector...]
    Y = Array{Float32}(undef, output_size, length(policy_vecs))
    for i in eachindex(policy_vecs)
        Y[:,i] = [returns[i], policy_vecs[i]...]
    end

    data = (X=X, Y=Y, G=G)

    return return_metrics ? (data, metrics) :  data
end


"""
Store generated data in the data buffer (separating training and validation split).
"""
function store_data!(solver::BetaZeroSolver, data)
    # Store data in buffer for training and validation
    # (separate the sets here so there is no chance of data leakage)
    n_data = size(data.Y,2)
    n_train = Int(n_data ÷ (1/solver.nn_params.training_split))
    perm = randperm(n_data) # shuffle data
    perm_train = perm[1:n_train]
    perm_valid = perm[n_train+1:n_data]
    x_size_span = map(d->1:d, solver.nn_params.input_size) # e.g., (1:30, 1:30, 1:5)

    X_train = getindex(data.X, x_size_span..., perm_train) # general for any size matrix e.g., x_train = x_data[:,:,:,perm_train]
    Y_train = data.Y[:, perm_train] # always assumed to be 1xN
    data_train = (X=X_train, Y=Y_train)
    push!(solver.data_buffer_train, data_train)

    X_valid = getindex(data.X, x_size_span..., perm_valid)
    Y_valid = data.Y[:, perm_valid]
    data_valid = (X=X_valid, Y=Y_valid)
    push!(solver.data_buffer_valid, data_valid)
end


"""
Compute the discounted `γ` returns from reward vector `R`.
"""
function compute_returns(R::Vector; γ::Real=1)
    T = length(R)
    G = zeros(T)
    for t in reverse(1:T)
        G[t] = t==T ? R[t] : G[t] = R[t] + γ*G[t+1]
    end
    return G
end


"""
Compute the predicted discounted returns using the current value network.
"""
function compute_predicted_returns(R::Vector, beliefs::Vector, network::Surrogate; γ::Real=1)
    T = length(R)
    G = zeros(T)
    for t in 1:T
        G[t] = R[t] + γ^(t)*value_lookup(network, beliefs[t])
    end
    return G
end


"""
Run single simulation using a belief-MCTS policy on the original POMDP (i.e., notabily, not on the belief-MDP).
"""
function run_simulation(pomdp::POMDP, policy::POMDPs.Policy, up::POMDPs.Updater, b0=initialize_belief(up, initialstate(pomdp)), s0=rand(b0);
                        max_steps=100,
                        ϵ=1e-8, # for policy vector
                        collect_metrics::Bool=false,
                        include_info::Bool=false,
                        show_time::Bool=false,
                        final_criterion::Any=MaxQN(),
                        use_completed_policy_gumbel::Bool=false,
                        skip_missing_reward_signal::Bool=false,
                        train_missing_on_predicted::Bool=false,
                        nn_params::Union{Nothing,BetaZeroNetworkParameters}=nothing)
    data = [BetaZeroTrainingData(b=input_representation(b0))]
    rewards::Vector{Float64} = [0.0]
    γ = POMDPs.discount(pomdp)
    action_space = POMDPs.actions(pomdp)

    local action
    local T
    infos::Vector = []
    beliefs::Vector = []
    states::Vector = [s0]
    actions::Vector = []

    include_info && push!(beliefs, b0)
    max_reached = false

    if train_missing_on_predicted
        if policy isa RawValueNetworkPolicy
            value_estimate = value_lookup(policy.surrogate, b0)
        else
            value_estimate = policy.solved_estimate(policy.mdp, b0, 0)
        end
        predicted_G = [rewards[1] + γ*value_estimate]
    end

    for (sp,a,r,bp,t,info) in stepthrough(pomdp, policy, up, b0, s0, "sp,a,r,bp,t,action_info", max_steps=max_steps)
        show_time && @info "Simulation: Time $t | Reward $r"
        T = t
        action = a
        push!(rewards, r)
        push!(data, BetaZeroTrainingData(b=input_representation(bp)))
        if include_info
            push!(infos, info)
            push!(beliefs, bp)
        end
        P = ϵ * ones(length(action_space))
        if !isnothing(info)
            if haskey(info, :counts)
                counts::Dict = info[:counts]
                root_actions = collect(keys(counts))
                root_counts_and_values = collect(values(counts))
                root_counts = first.(root_counts_and_values)
                root_values = last.(root_counts_and_values)
            elseif haskey(info, :completed_policy)
                completed_policy = info[:completed_policy]
            elseif haskey(info, :tree)
                tree = info[:tree]
                root_children_indices = tree.tried[1]
                root_actions = tree.a_labels[root_children_indices]
                root_counts = tree.n[root_children_indices]
                root_values = tree.v[root_children_indices]
            else
                error("Policy does not have root note visit information (or 'tree_in_info'/'counts_in_info' is not set)")
            end

            if use_completed_policy_gumbel
                P = completed_policy # Completed policy for guarenteed improvement (see Danihelka et al. 2022)
            else
                if final_criterion isa MaxQN
                    tree_P = normalize(softmax(root_values) .* root_counts, 1) # Q-weighted normalized counts (QWC)
                elseif final_criterion isa MaxN
                    tree_P = normalize(root_counts, 1) # Only use N(b,a) counts.
                elseif final_criterion isa MaxQ
                    tree_P = softmax(root_values) # Q-values
                elseif final_criterion isa SampleQN
                    τ = final_criterion.τ
                    QN = (softmax(root_values) .* root_counts) .^ (1/τ)
                    tree_P = normalize(QN, 1) # Exponentiated Q-weighted normalized counts (EQWC)
                elseif final_criterion isa MaxZQN
                    zq = final_criterion.zq
                    zn = final_criterion.zn
                    QN = (softmax(root_values).^zq .* root_counts.^zn)
                    tree_P = normalize(QN, 1)
                elseif final_criterion isa SampleZQN
                    τ = final_criterion.τ
                    zq = final_criterion.zq
                    zn = final_criterion.zn
                    QN = (softmax(root_values).^zq .* root_counts.^zn) .^ (1/τ)
                    tree_P = normalize(QN, 1)
                elseif final_criterion isa MaxWeightedQN
                    w = final_criterion.w
                    QN = (w*softmax(root_values) .+ (1-w)*root_counts)
                    tree_P = normalize(QN, 1) # TODO: Unnecessary, already normalized.
                elseif final_criterion isa SampleWeightedQN
                    τ = final_criterion.τ
                    w = final_criterion.w
                    QN = (w*softmax(root_values) .+ (1-w)*root_counts) .^ (1/τ)
                    tree_P = normalize(QN, 1) # Exponentiated Q-weighted normalized counts (EQWC)
                else
                    error("No policy data collection for the 'final_criterion' type of $(final_criterion)")
                end

                # Fill out entire policy vector for every action (if it wasn't seen in the tree, then p = ϵ for numerical stability)
                for (i,a′) in enumerate(action_space)
                    j = findfirst(tree_a->tree_a == a′, root_actions)
                    if !isnothing(j)
                        P[i] = tree_P[j]
                    end
                end
            end
        end

        if !isnothing(nn_params)
            if nn_params.use_dirichlet_exploration
                α = nn_params.α_dirichlet
                ϵ = nn_params.ϵ_dirichlet
                k = length(P)
                η = rand(Dirichlet(k, α))
                P = (1 - ϵ)*P + ϵ*η
            end
        end

        P = normalize(P, 1)
        data[end-1].π = P # associate policy vector with previous belief (i.e., belief node)
        push!(actions, a)
        push!(states, sp) # Note, initialized with s0

        if train_missing_on_predicted && iszero(rewards)
            # populate the missing reward signal with predicted returns
            if policy isa RawValueNetworkPolicy
                value_estimate = value_lookup(policy.surrogate, bp)
            else
                value_estimate = policy.solved_estimate(policy.mdp, bp, 0)
            end
            ṽ = r + γ*value_estimate
            push!(predicted_G, ṽ)
        end

        max_reached = (T == max_steps)
    end

    data[end].π = deepcopy(data[end-1].π) # terminal state, copy policy vector.

    G = compute_returns(rewards; γ=γ)
    real_returns = G

    if skip_missing_reward_signal && iszero(G) && max_reached
        # ignore cases were the time limit has been reached and no reward signal is present
        return missing, missing
    else
        if train_missing_on_predicted && iszero(G) && max_reached
            # populate the missing reward signal with predicted returns
            G = predicted_G
        end

        for (t,d) in enumerate(data)
            d.z = G[t]
        end
        metrics = collect_metrics ? compute_performance_metrics(pomdp, data, b0, s0, beliefs, states, actions, real_returns, infos, T) : nothing
        return data, metrics
    end
end


"""
Method to collect performance and validation metrics during BetaZero policy iteration.
Note, user defines `BetaZero.accuracy` to determine the accuracy of the final decision (if applicable).
"""
function compute_performance_metrics(pomdp::POMDP, data, b0, s0, beliefs, states, actions, returns, infos, T)
    # - mean discounted return over time
    # - accuracy over time (i.e., did it make the correct decision, if there's some notion of correct)
    # - number of actions (e.g., number of drills for mineral exploration)
    predicted_returns = [d.z for d in data]
    discounted_return = returns[1]
    optimal_G = BetaZero.optimal_return(pomdp, s0) # User defined per-POMDP
    accuracy = BetaZero.accuracy(pomdp, b0, s0, states, actions, returns) # Note: Problem specific, provide function to compute this.
    return (discounted_return=discounted_return, accuracy=accuracy, optimal_return=optimal_G, num_actions=T, infos=infos, beliefs=beliefs, actions=actions, predicted_returns=predicted_returns)
end


"""
Run a test on a holdout set to collect performance metrics during BetaZero policy iteration.
"""
function run_holdout_test!(pomdp::POMDP, solver::BetaZeroSolver, f::Surrogate; outer_iter::Int=0)
    if solver.params.n_holdout > 0
        solver.verbose && @info "Running holdout test..."
        data, metrics = generate_data(pomdp, solver, f; inner_iter=solver.params.n_holdout, outer_iter=outer_iter, holdout_criteria=true)
        returns = data.G
        accuracies = [m.accuracy for m in metrics]
        num_actions = [m.num_actions for m in metrics]
        optimal_returns = [m.optimal_return for m in metrics]
        try
            solver.verbose && display(UnicodePlots.histogram(returns))
        catch err
            @warn "Couldn't fit holdout histogram: $err"
        end
        μ, σ = mean_and_std(returns)
        push!(solver.holdout_metrics, (mean=μ, std=σ, returns=returns, accuracies=accuracies, num_actions=num_actions, optimal_returns=optimal_returns))
        solver.verbose && @show μ, σ

        if solver.plot_incremental_holdout
            performance_plot = plot_accuracy_and_returns(solver; include_holdout=true)
            solver.save_plots && Plots.savefig(solver.plot_metrics_filename)
            solver.display_plots && display(performance_plot)
        end
    end
end


"""
Check that the size of the data buffers and the `n_buffer` parameter agree.
If not, then resize the buffer based on the `n_buffer` parameter.
"""
function check_data_buffer_size!(solver::BetaZeroSolver)
    if capacity(solver.data_buffer_train) != solver.params.n_buffer
        @warn "Resizing data buffer to $(solver.params.n_buffer)"
        solver.data_buffer_train = CircularBuffer(solver.params.n_buffer)
        solver.data_buffer_valid = CircularBuffer(solver.params.n_buffer)
    end
end


"""
Get action from BetaZero policy (online MCTS using value & policy surrogate).
"""
POMDPs.action(policy::BetaZeroPolicy, b) = action(policy.planner, b)
POMDPTools.action_info(policy::BetaZeroPolicy, b; tree_in_info=false, counts_in_info=true) = POMDPTools.action_info(policy.planner, b; tree_in_info, counts_in_info)


end # module
