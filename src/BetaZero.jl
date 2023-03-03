module BetaZero

using BSON
using DataStructures
using Distributions
using Distributed
using Flux
using GaussianProcesses
using LinearAlgebra
using MCTS
using Optim
using Plots; default(fontfamily="Computer Modern", framestyle=:box)
using Parameters
using POMDPs
using POMDPTools
using ProgressMeter
using Random
using Statistics
using StatsBase
using Suppressor
using UnicodePlots

include("belief_mdp.jl")
include("representation.jl")
include("onestep_lookahead.jl")
include("bias.jl")
include("parameters.jl")

export
    BetaZeroSolver,
    BetaZeroPolicy,
    BetaZeroParameters,
    BetaZeroNetworkParameters,
    BetaZeroGPParameters,
    BeliefMDP,
    OneStepLookaheadSolver,
    RawNetworkPolicy,
    value_plot,
    policy_plot,
    value_and_policy_plot,
    plot_accuracy,
    plot_returns,
    plot_accuracy_and_returns


@with_kw mutable struct BetaZeroSolver <: POMDPs.Solver
    pomdp::POMDP
    updater::POMDPs.Updater
    params::BetaZeroParameters = BetaZeroParameters() # parameters for BetaZero algorithm
    nn_params::BetaZeroNetworkParameters = BetaZeroNetworkParameters(input_size=get_input_size(pomdp,updater), action_size=length(actions(pomdp))) # parameters for training NN
    gp_params::BetaZeroGPParameters = BetaZeroGPParameters(input_size=get_input_size(pomdp,updater)) # parameters for training GP
    data_buffer_train::CircularBuffer = CircularBuffer(params.n_buffer) # Simulation data buffer for training (NOTE: each simulation has multiple time steps of data)
    data_buffer_valid::CircularBuffer = CircularBuffer(params.n_buffer) # Simulation data buffer for validation (NOTE: making sure to clearly separate training from validation to prevent data leakage)
    bmdp::Union{BeliefMDP,Nothing} = nothing # Belief-MDP version of the POMDP
    belief_reward::Function = (pomdp::POMDP, b, a, bp)->0.0 # reward function: R(b,a,b′)
    # TODO: belief_representation::Function (see `representation.jl` TODO: should it be a parameter or overloaded function?)
    include_info::Bool = false # Include `action_info` in metrics when running POMDP simulation
    mcts_solver::AbstractMCTSSolver = DPWSolver(n_iterations=1000,
                                                check_repeat_action=true,
                                                exploration_constant=1.0,
                                                k_action=2.0,
                                                alpha_action=0.25,
                                                k_state=2.0,
                                                alpha_state=0.1,
                                                tree_in_info=true, # Note, required for policy vector.
                                                show_progress=false,
                                                estimate_value=(bmdp,b,d)->0.0) # `estimate_value` will be replaced with a surrogate lookup
    data_collection_policy::Policy = RandomPolicy(Random.GLOBAL_RNG, pomdp, updater) # Policy used for data collection (if indicated to use different policy than the BetaZero on-policy)
    use_data_collection_policy::Bool = false # Use provided policy for data collection.
    collect_metrics::Bool = true # Indicate that performance metrics should be collected.
    performance_metrics::Array = [] # Stored metrics for data generation runs.
    holdout_metrics::Array = [] # Metrics computed from holdout test set.
    accuracy_func::Function = (pomdp,belief,state,action,returns)->nothing # (returns Bool): Function to indicate that the decision was "correct" (if applicable)
    plot_incremental_data_gen::Bool = false # Plot accuracies and returns over iterations after data collection
    plot_incremental_holdout::Bool = false # Plot accuracies and returns over iterations after running holdout test
    expert_results::NamedTuple{(:expert_accuracy, :expert_returns, :expert_label), Tuple{Union{Nothing,Vector}, Union{Nothing,Vector}, String}} = (expert_accuracy=nothing, expert_returns=nothing, expert_label="expert") # For plotting comparisons, pass in the [mean, std] for the expert metrics.
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
include("ensamble.jl")
include("neural_network.jl")


const Surrogate = Union{Chain, GPSurrogate, EnsambleNetwork} # Needs GPSurrogate and EnsambleNetwork defined.

mutable struct BetaZeroPolicy <: POMDPs.Policy
    surrogate::Surrogate
    planner::AbstractMCTSPlanner
    parameters::ParameterCollection
end


include("raw_network.jl") # Needs Surrogate defined.
include("saving.jl") # Needs BetaZeroSolver and BetaZeroPolicy defined.
include("plots.jl")


"""
The main BetaZero policy iteration algorithm.
"""
function POMDPs.solve(solver::BetaZeroSolver, pomdp::POMDP; surrogate::Surrogate=solver.params.use_nn ? initialize_network(solver) : initialize_gaussian_process(solver))
    fill_bmdp!(pomdp, solver)
    f_prev::Surrogate = deepcopy(surrogate)

    @conditional_time solver.verbose for i in 1:solver.params.n_iterations
        solver.verbose && println(); println("—"^40); println(); @info "BetaZero iteration $i/$(solver.params.n_iterations)"

        # 0) Evaluate performance on a holdout test set (never used for training or surrogate selection).
        run_holdout_test!(pomdp, solver, f_prev)

        # 1) Generate data using the best BetaZero agent so far: {[belief, return], ...}
        use_different_policy = (i == 1) ? solver.use_data_collection_policy : false # only use data collection policy on first iteration
        generate_data!(pomdp, solver, f_prev; store_metrics=solver.collect_metrics, use_different_policy=use_different_policy, inner_iter=solver.params.n_data_gen, outer_iter=i)

        # 2) Optimize surrogate with recent simulated data (to estimate value given belief).
        f_curr = train(deepcopy(f_prev), solver; verbose=solver.verbose)

        # 3) Evaluate BetaZero agent (compare to previous agent based on mean returns).
        f_prev = evaluate_agent(pomdp, solver, f_prev, f_curr; outer_iter=typemax(Int32)+i)
    end

    # Re-run holdout test with final surrogate
    run_holdout_test!(pomdp, solver, f_prev)

    # Include the surrogate in the MCTS planner as part of the BetaZero policy
    return solve_planner!(solver, f_prev)
end


"""
Conver the `POMDP` to a `BeliefMDP` and set the `pomdp.bmdp` field.
"""
function fill_bmdp!(pomdp::POMDP, solver::BetaZeroSolver)
    solver.bmdp = BeliefMDP(pomdp, solver.updater, solver.belief_reward)
    return solver.bmdp
end


"""
Return the BetaZero planner, first adding the value estimator and then solving the inner MCTS planner.
"""
function solve_planner!(solver::BetaZeroSolver, f::Surrogate)
    attach_surrogate!(solver, f)
    mcts_planner = solve(solver.mcts_solver, solver.bmdp)
    parameters = ParameterCollection(solver.params, solver.nn_params, solver.gp_params)
    return BetaZeroPolicy(f, mcts_planner, parameters)
end


"""
Attach the surrogate model to the MCTS solver for value estimates and next action selection.
"""
function attach_surrogate!(solver::BetaZeroSolver, f::Surrogate)
    solver.mcts_solver.estimate_value = (bmdp,b,d)->value_lookup(b, f)
    solver.mcts_solver.next_action = (bmdp,b,bnode)->next_action(bmdp, b, f)
    return solver
end

function attach_surrogate!(policy::BetaZeroPolicy, f::Surrogate)
    policy.surrogate = f
    policy.planner.solver.estimate_value = (bmdp,b,d)->value_lookup(b, f)
    policy.planner.solver.next_action = (bmdp,b,bnode)->next_action(bmdp, b, f)
    policy.planner = solve(policy.planner.solver, policy.planner.mdp) # Important for online policy
    return policy
end


"""
Uniformly sample data from buffer (with replacement).
Note that the buffer is per-simulation with each simulation having multiple time steps.
We want to sample `n` individual time steps across the simulations.
"""
function sample_data(data_buffer::CircularBuffer, n::Int)
    sim_times = map(d->size(d.Y,2), data_buffer) # number of time steps in each simulation
    data_buffer_indices = 1:length(data_buffer)
    sampled_sims_indices = sample(data_buffer_indices, Weights(sim_times), n; replace=true) # weighted based on num. steps per sim (to keep with __overall__ uniform across time steps)
    belief_size = size(data_buffer[1].X)[1:end-1]
    X = Array{Float32}(undef, belief_size..., n)
    output_size = size(data_buffer[1].Y, 1)
    Y = Array{Float32}(undef, output_size, n)
    for (i,sim_i) in enumerate(sampled_sims_indices)
        sim = data_buffer[sim_i]
        T = size(sim.Y, 2)
        t = rand(1:T) # uniformly sample time from this simulation
        belief_size_span = map(d->1:d, belief_size) # e.g., (1:30, 1:30, 1:5)
        setindex!(X, getindex(sim.X, belief_size_span..., t), belief_size_span..., i) # general for any size matrix e.g., X[:,;,:,i] = sim.X[:,:,:,t]
        Y[:, i] = sim.Y[:, t]
    end
    return (X=X, Y=Y)
end


"""
Compare previous and current neural networks using MCTS simulations.
Use upper confidence bound on the discounted return as the comparison metric.
"""
function evaluate_agent(pomdp::POMDP, solver::BetaZeroSolver, f_prev::Surrogate, f_curr::Surrogate; outer_iter::Int=0)
    # Run a number of simulations to evaluate the two neural networks using MCTS (`f_prev` and `f_curr`)
    if solver.params.n_evaluate == 0
        solver.verbose && @info "Skipping surrogate evaluations, selected newest surrogate."
        return f_curr
    else
        solver.verbose && @info "Evaluting networks..."
        returns_prev = generate_data!(pomdp, solver, f_prev; inner_iter=solver.params.n_evaluate, outer_iter=outer_iter, store_data=false)[:G]
        returns_curr = generate_data!(pomdp, solver, f_curr; inner_iter=solver.params.n_evaluate, outer_iter=outer_iter, store_data=false)[:G]

        λ = solver.params.λ_ucb
        μ_prev, σ_prev = mean_and_std(returns_prev)
        μ_curr, σ_curr = mean_and_std(returns_curr)
        ucb_prev = μ_prev + λ*σ_prev
        ucb_curr = μ_curr + λ*σ_curr

        solver.verbose && @show ucb_curr, ucb_prev

        if ucb_curr > ucb_prev
            solver.verbose && @info "<<<< New surrogate performed better >>>>"
            return f_curr
        else
            if solver.verbose && ucb_curr == ucb_prev
                @info "[IDENTICAL UCBs]"
            end
            solver.verbose && @info "---- Previous surrogate performed better ----"
            return f_prev
        end
    end
end


"""
Generate training data using online MCTS with the best surrogate so far `f` (parallelized across episodes).
"""
function generate_data!(pomdp::POMDP, solver::BetaZeroSolver, f::Surrogate;
                        outer_iter::Int=0, inner_iter::Int=solver.params.n_data_gen,
                        store_metrics::Bool=false, store_data::Bool=true,
                        return_metrics::Bool=false,
                        use_different_policy::Bool=false)
    # Confirm that surrogate is on the CPU for inference
    f = cpu(f)
    up = solver.updater
    isnothing(solver.bmdp) && fill_bmdp!(pomdp, solver)

    if use_different_policy
        @info "Using provided policy for data generation..."
        planner = solver.data_collection_policy
    else
        # Run MCTS to generate data using the surrogate `f`
        attach_surrogate!(solver, f)
        planner = solve(solver.mcts_solver, solver.bmdp)
    end

    ds0 = POMDPs.initialstate_distribution(pomdp)
    collect_metrics = solver.collect_metrics
    accuracy_func = solver.accuracy_func
    include_info = solver.include_info

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
            s0 = rand(ds0)
            b0 = POMDPs.initialize_belief(up, ds0)
            data, metrics = run_simulation(pomdp, planner, up, b0, s0; collect_metrics, accuracy_func, include_info)
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

    if store_metrics
        push!(solver.performance_metrics, metrics...)
    end

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

    if store_data
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

        # Plot incremental learning
        if solver.plot_incremental_data_gen
            plot_accuracy_and_returns(solver; include_holdout=solver.plot_incremental_holdout) |> display
        end
    end

    return return_metrics ? (data, metrics) :  data
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
Run single simulation using a belief-MCTS policy on the original POMDP (i.e., notabily, not on the belief-MDP).
"""
function run_simulation(pomdp::POMDP, policy::POMDPs.Policy, up::POMDPs.Updater, b0, s0;
                        max_steps=100, collect_metrics::Bool=false, accuracy_func::Function=(args...)->nothing, include_info::Bool=false)
    rewards::Vector{Float64} = [0.0]
    data = [BetaZeroTrainingData(b=input_representation(b0))]
    local action
    local T
    infos::Vector = []
    beliefs::Vector = []
    actions::Vector = []

    include_info && push!(beliefs, b0)
    max_reached = false

    action_space = POMDPs.actions(pomdp)

    for (a,r,bp,t,info) in stepthrough(pomdp, policy, up, b0, s0, "a,r,bp,t,action_info", max_steps=max_steps)
        # @info "Simulation time step $t"
        T = t
        action = a
        push!(rewards, r)
        push!(data, BetaZeroTrainingData(b=input_representation(bp)))
        if include_info
            push!(infos, info)
            push!(beliefs, bp)
        end
        if !isnothing(info) && haskey(info, :tree)
            tree = info[:tree]
            root_idx = first(tree.children)
            tree_A = tree.a_labels[root_idx]
            tree_N = tree.n[root_idx]
            tree_P = normalize(tree_N, 1)
            P = zeros(length(action_space))
            # Fill out entire policy vector for every action (if it wasn't seen in the tree, then p=0)
            for (i,a′) in enumerate(action_space)
                j = findfirst(tree_a->tree_a == a′, tree_A)
                if !isnothing(j)
                    P[i] = tree_P[j]
                end
            end
            data[end-1].π = P # associate policy vector with previous belief (i.e., belief node)
        end
        push!(actions, a)
        max_reached = (T == max_steps)
    end

    data[end].π = deepcopy(data[end-1].π) # terminal state, copy policy vector.

    γ = POMDPs.discount(pomdp)
    G = compute_returns(rewards; γ=γ)

    skip_missing_reward_signal = false # NOTE.

    if skip_missing_reward_signal && iszero(G) && max_reached
        # ignore cases were the time limit has been reached and no reward signal is present
        return missing, missing
    else
        for (t,d) in enumerate(data)
            d.z = G[t]
        end
        metrics = collect_metrics ? compute_performance_metrics(pomdp, data, accuracy_func, b0, s0, beliefs, actions, infos, T) : nothing
        return data, metrics
    end
end


"""
Method to collect performance and validation metrics during BetaZero policy iteration.
Note, user defines `solver.accuracy_func` to determine the accuracy of the final decision (if applicable).
"""
function compute_performance_metrics(pomdp::POMDP, data, accuracy_func::Function, b0, s0, beliefs, actions, infos, T)
    # - mean discounted return over time
    # - accuracy over time (i.e., did it make the correct decision, if there's some notion of correct)
    # - number of actions (e.g., number of drills for mineral exploration)
    returns = [d.z for d in data]
    discounted_return = returns[1]
    final_action = actions[end]
    accuracy = accuracy_func(pomdp, b0, s0, final_action, returns) # NOTE: Problem specific, provide function to compute this
    return (discounted_return=discounted_return, accuracy=accuracy, num_actions=T, infos=infos, beliefs=beliefs, actions=actions)
end


"""
Run a test on a holdout set to collect performance metrics during BetaZero policy iteration.
"""
function run_holdout_test!(pomdp::POMDP, solver::BetaZeroSolver, f::Surrogate; outer_iter::Int=0)
    if solver.params.n_holdout > 0
        solver.verbose && @info "Running holdout test..."
        data, metrics = generate_data!(pomdp, solver, f;
                                 inner_iter=solver.params.n_holdout, outer_iter=outer_iter,
                                 store_metrics=false, store_data=false, return_metrics=true)
        returns = data.G
        accuracies = [m.accuracy for m in metrics]
        num_actions = [m.num_actions for m in metrics]
        try
            solver.verbose && display(UnicodePlots.histogram(returns))
        catch err
            @warn "Couldn't fit holdout histogram: $err"
        end
        μ, σ = mean_and_std(returns)
        push!(solver.holdout_metrics, (mean=μ, std=σ, returns=returns, accuracies=accuracies, num_actions=num_actions))
        solver.verbose && @show μ, σ

        if solver.plot_incremental_holdout
            plot_accuracy_and_returns(solver; include_holdout=true) |> display
        end
    end
end


"""
Get action from BetaZero policy (online MCTS using value & policy surrogate).
"""
POMDPs.action(policy::BetaZeroPolicy, b) = action(policy.planner, b)
POMDPTools.action_info(policy::BetaZeroPolicy, b; tree_in_info=false) = POMDPTools.action_info(policy.planner, b; tree_in_info)


end # module
