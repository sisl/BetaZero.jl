module BetaZero

using BeliefUpdaters
using BSON
using DataStructures
using Distributed
using Flux
using MCTS
using Plots; default(fontfamily="Computer Modern", framestyle=:box)
using Parameters
using POMDPs
using POMDPTools
using ProgressMeter
using Random
using Statistics
using StatsBase
using UnicodePlots

include("belief_mdp.jl")
include("representation.jl")
include("onestep_lookahead.jl")

export
    BetaZeroSolver,
    BetaZeroPolicy,
    BetaZeroNetworkParameters,
    BeliefMDP,
    OneStepLookaheadSolver


mutable struct BetaZeroPolicy <: POMDPs.Policy
    network::Chain
    planner::AbstractMCTSPlanner
end


@with_kw mutable struct BetaZeroNetworkParameters
    input_size = (30,30,5)
    training_epochs::Int = 1000 # Number of network training updates
    n_samples::Int = 10_000 # Number of samples (i.e., simulated POMDP time steps from data collection) to use during training + validation
    normalize_target::Bool = true # Normalize target data to standard normal (0 mean)
    training_split::Float64 = 0.8 # Training / validation split (Default: 80/20)
    batchsize::Int = 512
    learning_rate::Float64 = 0.01 # Learning rate for ADAM optimizer during training
    λ_regularization::Float64 = 0.0001 # Parameter for L2-norm regularization
    loss_func::Function = Flux.Losses.mae # MAE works well for problems with large returns around zero, and spread out otherwise.
    device = gpu
    verbose_update_frequency::Int = training_epochs # Frequency of printed training output
    verbose_plot_frequency::Number = 10 # Frequency of plotted training/validation output
end


@with_kw mutable struct BetaZeroSolver <: POMDPs.Solver
    pomdp::POMDP
    n_iterations::Int = 1 # BetaZero policy iterations (primary outer loop).
    n_data_gen::Int = 10 # Number of episodes to run for training/validation data generation.
    n_evaluate::Int = 0 # Number of episodes to run for network evaluation and comparison.
    n_holdout::Int = 10 # Number of episodes to run for a holdout test set (on a fixed, non-training or evaluation set).
    n_buffer::Int = n_iterations # Number of iterations to keep data for network training (NOTE: each simulation has multiple time steps of data, not counted in this number. This number corresponds to the number of iterations, i.e., set to 2 if you want to keep data from the previous 2 policy iterations.)
    data_buffer::CircularBuffer = CircularBuffer(n_buffer) # Simulation data buffer for training (NOTE: each simulation has multiple time steps of data)
    λ_ucb::Real = 0.0 # Upper confidence bound parameter: μ + λσ # TODO: Remove?
    updater::POMDPs.Updater
    network_params::BetaZeroNetworkParameters = BetaZeroNetworkParameters(input_size=size(BetaZero.input_representation(initialize_belief(updater, initialstate(pomdp))))) # parameters for training CNN
    belief_reward::Function = (pomdp::POMDP, b, a, bp)->0.0
    # TODO: belief_representation::Function (see `representation.jl` TODO: should it be a parameter or overloaded function?)
    include_info::Bool = false # Include `action_info` in metrics when running POMDP simulation
    mcts_solver::AbstractMCTSSolver = DPWSolver(n_iterations=10,
                                                check_repeat_action=true,
                                                exploration_constant=1.0, # 1.0
                                                k_action=2.0, # 10
                                                alpha_action=0.25, # 0.5
                                                k_state=2.0, # 10
                                                alpha_state=0.1, # 0.5
                                                tree_in_info=false,
                                                show_progress=false,
                                                estimate_value=(bmdp,b,d)->0.0) # `estimate_value` will be replaced with a neural network lookup
    onestep_solver::OneStepLookaheadSolver = OneStepLookaheadSolver(n_actions=5,
                                                                    n_obs=2,
                                                                    estimate_value=b->0.0)
    data_gen_policy = nothing
    use_onestep_lookahead_holdout::Bool = true # Use greedy one-step lookahead solver when checking performance on the holdout set
    use_random_policy_data_gen::Bool = true # Use random policy for data generation
    bmdp::Union{BeliefMDP,Nothing} = nothing # Belief-MDP version of the POMDP
    collect_metrics::Bool = true # Indicate that performance metrics should be collected.
    performance_metrics::Array = [] # TODO: store_metrics for NON-HOLDOUT runs.
    holdout_metrics::Array = [] # Metrics computed from holdout test set.
    accuracy_func::Function = (pomdp,belief,state,action,returns)->nothing # (returns Bool): Function to indicate that the decision was "correct" (if applicable)
    verbose::Bool = true # Print out debugging/training/simulation information during solving
end


@with_kw mutable struct BetaZeroTrainingData
    b = nothing # current belief
    π = nothing # current policy estimate (using N(s,a))
    z = nothing # final discounted return of the episode
end


# Needs BetaZeroSolver defined.
include("metrics.jl")


"""
Run @time on expression based on `verbose` flag.
"""
macro conditional_time(verbose, expr)
    esc(quote
        if $verbose
            @time $expr
        else
            $expr
        end
    end)
end


"""
The main BetaZero policy iteration algorithm.
"""
function POMDPs.solve(solver::BetaZeroSolver, pomdp::POMDP)
    fill_bmdp!(pomdp, solver)
    f_prev = initialize_network(solver)

    @conditional_time solver.verbose for i in 1:solver.n_iterations
        solver.verbose && println(); println("—"^40); println(); @info "BetaZero iteration $i/$(solver.n_iterations)"

        # 0) Evaluate performance on a holdout test set (never used for training or network selection).
        # run_holdout_test!(pomdp, solver, f_prev; outer_iter=i) # TODO: DEBUGGING
        run_holdout_test!(pomdp, solver, f_prev)

        # 1) Generate data using the best BetaZero agent so far: {[belief, return], ...}
        generate_data!(pomdp, solver, f_prev; use_random_policy=solver.use_random_policy_data_gen, inner_iter=solver.n_data_gen, outer_iter=i)

        # 2) Optimize neural network parameters with recent simulated data (to estimate value given belief).
        f_curr = train_network(deepcopy(f_prev), solver; verbose=solver.verbose)

        # 3) Evaluate BetaZero agent (compare to previous agent based on mean returns).
        # f_prev = evaluate_agent(pomdp, solver, f_prev, f_curr; outer_iter=i) # TODO: DEBUGGING
        f_prev = evaluate_agent(pomdp, solver, f_prev, f_curr; outer_iter=typemax(Int32)+i)
    end

    if solver.n_iterations == 1
        # Re-run holdout test if only running for a single iteration
        run_holdout_test!(pomdp, solver, f_prev)
    end

    solver.mcts_solver.estimate_value = (bmdp,b,d)->value_lookup(b, f_prev)
    mcts_planner = solve(solver.mcts_solver, solver.bmdp)
    policy = BetaZeroPolicy(f_prev, mcts_planner)

    return policy
end


"""
Conver the `POMDP` to a `BeliefMDP` and set the `pomdp.bmdp` field.
"""
function fill_bmdp!(pomdp::POMDP, solver::BetaZeroSolver)
    solver.bmdp = BeliefMDP(pomdp, solver.updater, solver.belief_reward)
    return solver.bmdp
end


"""
Initialize policy & value network with random weights.
"""
initialize_network(solver::BetaZeroSolver) = initialize_network(solver.network_params)
function initialize_network(nn_params::BetaZeroNetworkParameters) # LeNet5
    input_size = nn_params.input_size
    filter = (5,5)
    num_filters1 = 6
    num_filters2 = 16
    out_conv_size = prod([input_size[1] - 2*(filter[1]-1), input_size[2] - 2*(filter[2]-1), num_filters2])
    num_dense1 = 120
    num_dense2 = 84
    out_dim = 1

    return Chain(
        Conv(filter, input_size[end]=>num_filters1, relu),
        Conv(filter, num_filters1=>num_filters2, relu),
        Flux.flatten,
        Dense(out_conv_size, num_dense1, relu),
        Dense(num_dense1, num_dense2, relu),
        Dense(num_dense2, out_dim),
        # Note: A normalization layer will be added during training (with the old layer removed before the next training phase).
    )
end


"""
Train policy & value neural network `f` using the latest `data` generated from online tree search (MCTS).
"""
function train_network(f, solver::BetaZeroSolver; verbose::Bool=false, results=nothing)
    nn_params = solver.network_params
    lr = nn_params.learning_rate
    λ = nn_params.λ_regularization
    loss_str = string(nn_params.loss_func)
    normalize_target = nn_params.normalize_target
    key = (lr, λ, loss_str, normalize_target)

    data = sample_data(solver.data_buffer, nn_params.n_samples) # sample `n_samples` from last `n_buffer` simulations.
    x_data, y_data = data.X, data.Y

    # Normalize target values close to the range of [-1, 1]
    if normalize_target
        mean_y = mean(y_data)
        std_y = std(y_data)
        y_data = (y_data .- mean_y) ./ std_y
    end

    n_data = length(y_data)
    n_train = Int(n_data ÷ (1/nn_params.training_split))

    verbose && @info "Data set size: $n_data"

    perm = randperm(n_data)
    perm_train = perm[1:n_train]
    perm_valid = perm[n_train+1:n_data]

    x_size_span = map(d->1:d, nn_params.input_size) # e.g., (1:30, 1:30, 1:5)
    x_train = getindex(x_data, x_size_span..., perm_train) # general for any size matrix e.g., x_train = x_data[:,:,:,perm_train]
    y_train = y_data[:, perm_train] # always assumed to be 1xN (TODO: Changes when dealing with policy vector)

    x_valid = getindex(x_data, x_size_span..., perm_valid) # general for any size matrix e.g., x_valid = x_data[:,:,:,perm_valid]
    y_valid = y_data[:, perm_valid]

    # Put model/data onto GPU device
    device = nn_params.device
    x_train = device(x_train)
    y_train = device(y_train)
    x_valid = device(x_valid)
    y_valid = device(y_valid)

    if n_train < nn_params.batchsize
        batchsize = n_train
        @warn "Number of observations less than batch-size, decreasing the batch-size to $batchsize"
    else
        batchsize = nn_params.batchsize
    end

    train_data = Flux.Data.DataLoader((x_train, y_train), batchsize=batchsize, shuffle=true)

    # Remove un-normalization layer (if added from previous iteration)
    # We want to train for values close to [-1, 1]
    if isa(f.layers[end], Function) && normalize_target
        f = Chain(f.layers[1:end-1]...)
    end

    # Put network on GPU for training
    f = device(f)

    # TODO: Include action/policy vector and change loss to include CE-loss

    sqnorm(x) = sum(abs2, x)
    penalty() = λ*sum(sqnorm, Flux.params(f))
    sign_accuracy(x, y) = mean(sign.(f(x)) .== sign.(y)) # TODO: Generalize
    loss(x, y) = nn_params.loss_func(f(x), y) + penalty()

    opt = Adam(lr)
    θ = Flux.params(f)

    training_epochs = nn_params.training_epochs
    losses_train = []
    losses_valid = []
    accs_train = []
    accs_valid = []
    verbose && @info "Beginning training $(size(x_train))"
    @conditional_time verbose for e in 1:training_epochs
        for (x, y) in train_data
            _, back = Flux.pullback(() -> loss(x, y), θ)
            Flux.update!(opt, θ, back(1.0f0))
        end
        loss_train = loss(x_train, y_train)
        loss_valid = loss(x_valid, y_valid)
        acc_train = sign_accuracy(x_train, y_train)
        acc_valid = sign_accuracy(x_valid, y_valid)
        push!(losses_train, loss_train)
        push!(losses_valid, loss_valid)
        push!(accs_train, acc_train)
        push!(accs_valid, acc_valid)
        if verbose && e % nn_params.verbose_update_frequency == 0
            println("Epoch: ", e, " Loss Train: ", loss_train, " Loss Val: ", loss_valid, " | Acc. Train: ", acc_train, " Acc. Val: ", acc_valid)
        end
        if e % nn_params.verbose_plot_frequency == 0
            # TODO: Generalize
            plot(xlims=(1, training_epochs), ylims=(0, 1), title="learning curve: $key")
            plot!(1:e, losses_train, label="training")
            plot!(1:e, losses_valid, label="validation")
            display(plot!())
        end
    end

    # Check value distributions of model and data
    value_model = cpu(f(x_valid))'
    value_data = cpu(y_valid)'
    if normalize_target
        value_model = (value_model .* std_y) .+ mean_y
        value_data = (value_data .* std_y) .+ mean_y
    end

    if nn_params.verbose_plot_frequency != Inf
        learning_curve = nothing
        value_distribution = nothing
        try
            learning_curve = plot!()
            display(learning_curve)
            value_distribution = Plots.histogram(value_model, alpha=0.5, label="model", c=:gray, title="values: $key")
            Plots.histogram!(value_data, alpha=0.5, label="data", c=:navy)
            display(value_distribution)
        catch err
            @warn "Error in plotting learning curve and value distribution: $err"
        end
    end

    # Save training results
    if !isnothing(results) && isa(results, Dict)
        results[key] = Dict(
            "losses_train" => losses_train,
            "losses_valid" => losses_valid,
            "accs_train" => accs_train,
            "accs_valid" => accs_valid,
            "value_model" => value_model,
            "value_data" => value_data,
            "curve" => learning_curve,
            "value_distribution" => value_distribution,
        )

        if nn_params.verbose_plot_frequency != Inf
            results[key]["curve"] = learning_curve
            results[key]["value_distribution"] = value_distribution
        end
    end

    # Place network on the CPU (better GPU memory conservation when doing parallelized inference)
    f = cpu(f)

    # Clean GPU memory explicitly
    if device == gpu
        x_train = y_train = x_valid = y_valid = nothing
        GC.gc()
        Flux.CUDA.reclaim()
    end

    if normalize_target
        # Add un-normalization layer
        unnormalize = y -> (y .* std_y) .+ mean_y
        f = Chain(f.layers..., unnormalize)
    end

    return f
end


"""
Evaluate the neural network `f` using the `belief` as input.
Note, inference is done on the CPU given a single input.
"""
function value_lookup(belief, f)
    b = input_representation(belief)
    b = Float32.(b)
    x = Flux.unsqueeze(b; dims=ndims(b)+1)
    y = f(x) # evaluate network `f`
    value = cpu(y)[1] # returns 1 element 1D array
    return value
end



"""
Uniformly sample data from buffer (with replacement).
Note that the buffer is per-simulation with each simulation having multiple time steps.
We want to sample `n` individual time steps across the simulations.
"""
function sample_data(data_buffer::CircularBuffer, n::Int)
    sim_times = map(d->length(d.Y), data_buffer) # number of time steps in each simulation
    data_buffer_indices = 1:length(data_buffer)
    sampled_sims_indices = sample(data_buffer_indices, Weights(sim_times), n; replace=true) # weighted based on num. steps per sim (to keep with __overall__ uniform across time steps)
    belief_size = size(data_buffer[1].X)[1:end-1]
    X = Array{Float32}(undef, belief_size..., n)
    Y = Array{Float32}(undef, 1, n)
    G = Vector{Float32}(undef, n)
    for (i,sim_i) in enumerate(sampled_sims_indices)
        sim = data_buffer[sim_i]
        T = length(sim.Y)
        t = rand(1:T) # uniformly sample time from this simulation
        belief_size_span = map(d->1:d, belief_size) # e.g., (1:30, 1:30, 1:5)
        setindex!(X, getindex(sim.X, belief_size_span..., t), belief_size_span..., i) # general for any size matrix e.g., X[:,;,:,i] = sim.X[:,:,:,t]
        Y[i] = sim.Y[t]
        G[i] = sim.G[sim_i]
    end
    return (X=X, Y=Y, G=G)
end


"""
Compare previous and current neural networks using MCTS simulations.
Use upper confidence bound on the discounted return as the comparison metric.
"""
function evaluate_agent(pomdp::POMDP, solver::BetaZeroSolver, f_prev, f_curr; outer_iter=0)
    # Run a number of simulations to evaluate the two neural networks using MCTS (`f_prev` and `f_curr`)
    if solver.n_evaluate == 0
        solver.verbose && @info "Skipping network evaluations, selected newest network."
        return f_curr
    else
        solver.verbose && @info "Evaluting networks..."
        returns_prev = generate_data!(pomdp, solver, f_prev; inner_iter=solver.n_evaluate, outer_iter=outer_iter, store_data=false)[:G]
        returns_curr = generate_data!(pomdp, solver, f_curr; inner_iter=solver.n_evaluate, outer_iter=outer_iter, store_data=false)[:G]

        λ = solver.λ_ucb
        μ_prev, σ_prev = mean_and_std(returns_prev)
        μ_curr, σ_curr = mean_and_std(returns_curr)
        ucb_prev = μ_prev + λ*σ_prev
        ucb_curr = μ_curr + λ*σ_curr

        solver.verbose && @show ucb_curr, ucb_prev

        if ucb_curr > ucb_prev
            solver.verbose && @info "<<<< New network performed better >>>>"
            return f_curr
        else
            if solver.verbose && ucb_curr == ucb_prev
                @info "[IDENTICAL UCBs]"
            end
            solver.verbose && @info "---- Previous network performed better ----"
            return f_prev
        end
    end
end


"""
Generate training data using online MCTS with the best network so far `f` (parallelized across episodes).
"""
function generate_data!(pomdp::POMDP, solver::BetaZeroSolver, f; outer_iter::Int=0, inner_iter::Int=solver.n_data_gen, store_metrics::Bool=false, store_data::Bool=true, use_onestep_lookahead::Bool=false, use_random_policy::Bool=false)
    # Confirm that network is on the CPU for inference
    f = cpu(f)
    up = solver.updater
    isnothing(solver.bmdp) && fill_bmdp!(pomdp, solver)

    if use_random_policy
        @info "Using random policy for data generation..."
        planner = RandomPolicy(Random.GLOBAL_RNG, pomdp, up)
        # @info "Using provided heuristic policy for data generation..."
        # planner = solver.data_gen_policy
    elseif use_onestep_lookahead
        # Use greedy one-step lookahead with neural network `f`
        solver.onestep_solver.estimate_value = b->value_lookup(b, f)
        planner = solve(solver.onestep_solver, solver.bmdp)
    else
        # Run MCTS to generate data using the neural network `f`
        solver.mcts_solver.estimate_value = (bmdp,b,d)->value_lookup(b, f)
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
            B = []
            Z = []
            # Π = []
            discounted_return = data[1].z
            for d in data
                push!(B, d.b)
                push!(Z, d.z)
                # push!(Π, d.π) # TODO.
            end
            put!(channel, true) # trigger progress bar update
            B, Z, metrics, discounted_return
        end, 1:inner_iter)

    put!(channel, false) # tell printing task to finish

    beliefs = vcat([d[1] for d in parallel_data]...) # combine all beliefs
    returns = vcat([d[2] for d in parallel_data]...) # combine all returns
    metrics = vcat([d[3] for d in parallel_data]...) # combine all metrics
    G = vcat([d[4] for d in parallel_data]...) # combine all final returns

    if store_metrics
        push!(solver.performance_metrics, metrics...)
    end

    # Much faster than `cat(belief...; dims=4)`
    belief = beliefs[1]
    X = Array{Float32}(undef, size(belief)..., length(beliefs))
    for i in eachindex(beliefs)
        # Generalize for any size matrix (equivalent to X[:,:,:,i] = beliefs[i] for 3D matrix)
        setindex!(X, beliefs[i], map(d->1:d, size(belief))..., i)
    end
    Y = reshape(Float32.(returns), 1, length(returns))

    data = (X=X, Y=Y, G=G)

    if store_data
        # Store data in buffer for training
        push!(solver.data_buffer, data)
    end

    return data
end


"""
Compute the discounted `γ` returns from reward vector `R`.
"""
function compute_returns(R; γ=1)
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
        push!(actions, a)
    end

    γ = POMDPs.discount(pomdp)
    G = compute_returns(rewards; γ=γ)

    for (t,d) in enumerate(data)
        d.z = G[t]
    end

    metrics = collect_metrics ? compute_performance_metrics(pomdp, data, accuracy_func, b0, s0, beliefs, actions, infos, T) : nothing

    return data, metrics
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
function run_holdout_test!(pomdp::POMDP, solver::BetaZeroSolver, f; outer_iter::Int=0)
    if solver.n_holdout > 0
        solver.verbose && @info "Running holdout test..."
        returns = generate_data!(pomdp, solver, f;
                                 inner_iter=solver.n_holdout, outer_iter=outer_iter,
                                 store_metrics=true, store_data=false,
                                 use_onestep_lookahead=solver.use_onestep_lookahead_holdout)[:G]
        solver.verbose && display(UnicodePlots.histogram(returns))
        μ, σ = mean_and_std(returns)
        push!(solver.holdout_metrics, (mean=μ, std=σ, returns=returns))
        solver.verbose && @show μ, σ
    end
end


"""
Save performance metrics to a file.
"""
function save_metrics(solver::BetaZeroSolver, filename::String)
    metrics = solver.performance_metrics
    BSON.@save filename metrics
end


"""
Save policy to file (MCTS planner and network objects together).
"""
function save_policy(policy::BetaZeroPolicy, filename::String)
    BSON.@save "$filename" policy
end


"""
Load policy from file (MCTS planner and network objects together).
"""
function load_policy(filename::String)
    BSON.@load "$filename" policy
    return policy
end


"""
Save just the neural network to a file.
"""
function save_network(policy::BetaZeroPolicy, filename::String)
    network = policy.network
    BSON.@save "$filename" network
end


"""
Load just the neural network from a file.
"""
function load_network(filename::String)
    BSON.@load "$filename" network
    return network
end


"""
Get action from BetaZero policy (online MCTS using value & policy network).
"""
POMDPs.action(policy::BetaZeroPolicy, b) = action(policy.planner, b)
POMDPTools.action_info(policy::BetaZeroPolicy, b; tree_in_info=false) = POMDPTools.action_info(policy.planner, b; tree_in_info)


end # module
