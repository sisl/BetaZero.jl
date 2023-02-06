module BetaZero

using BSON
using Distributed
using Flux
using MCTS
using Plots; default(fontfamily="Computer Modern", framestyle=:box)
using Parameters
using POMDPs
using POMDPSimulators
using ProgressMeter
using Random
using Statistics
using StatsBase
using UnicodePlots

include("belief_mdp.jl")
include("representation.jl")

export
    BetaZeroSolver,
    BetaZeroPolicy,
    BetaZeroNetworkParameters,
    BeliefMDP


mutable struct BetaZeroPolicy <: POMDPs.Policy
    network::Chain
    planner::AbstractMCTSPlanner
end


@with_kw mutable struct BetaZeroNetworkParameters
    input_size = (30,30,5)
    training_epochs = 1000
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
    n_iterations::Int = 10 # BetaZero policy iterations (primary outer loop).
    n_data_gen::Int   = 100 # Number of episodes to run for training/validation data generation.
    n_evaluate::Int   = 25 # Number of episodes to run for network evaluation and comparison.
    n_holdout::Int    = 10 # Number of episodes to run for a holdout test set (on a fixed, non-training or evaluation set).
    λ_ucb::Real       = 0.0 # Upper confidence bound parameter: μ + λσ
    updater::POMDPs.Updater
    network_params::BetaZeroNetworkParameters = BetaZeroNetworkParameters() # parameters for training CNN
    belief_reward::Function = (pomdp::POMDP, b, a, bp)->0.0
    # belief representation function (see `representation.jl` TODO: should it be a parameter or overloaded function?)
    mcts_solver::AbstractMCTSSolver = DPWSolver(n_iterations=100,
                                                check_repeat_action=true,
                                                exploration_constant=1.0, # 10.0
                                                k_action=10.0, # 2
                                                alpha_action=0.5, # 0.25
                                                k_state=10.0, # 2
                                                alpha_state=0.5, # 0.25
                                                tree_in_info=false,
                                                show_progress=false,
                                                estimate_value=(bmdp,b,d)->0.0) # `estimate_value` will be replaced with a neural network lookup
    bmdp::Union{BeliefMDP,Nothing} = nothing # Belief-MDP version of the POMDP
    collect_metrics::Bool = true # Indicate that performance metrics should be collected.
    performance_metrics::Array = []
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
        run_holdout_test!(pomdp, solver, f_prev)

        # 1) Generate data using the best BetaZero agent so far: {[belief, return], ...}
        data = generate_data(pomdp, solver, f_prev; outer_iter=i)

        # 2) Optimize neural network parameters with recent simulated data (to estimate value given belief).
        f_curr = train_network(deepcopy(f_prev), data, solver.network_params; verbose=solver.verbose)

        # 3) Evaluate BetaZero agent (compare to previous agent based on mean returns).
        f_prev = evaluate_agent(pomdp, solver, f_prev, f_curr; outer_iter=typemax(Int32)+i)
    end

    solver.mcts_solver.estimate_value = (bmdp,b,d)->value_lookup(solver.network_params, b, f_prev)
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
function initialize_network(solver::BetaZeroSolver) # LeNet5
    nn_params = solver.network_params
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
function train_network(f, data, nn_params::BetaZeroNetworkParameters; verbose::Bool=false)
    x_data, y_data = data.X, data.Y

    # Normalize target values close to the range of [-1, 1]
    if nn_params.normalize_target
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

    x_train = x_data[:,:,:,perm_train]
    y_train = y_data[:, perm_train]

    x_valid = x_data[:,:,:,perm_valid]
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
    if isa(f.layers[end], Function)
        f = Chain(f.layers[1:end-1]...)
    end

    # Put network on GPU for training
    f = device(f)

    sqnorm(x) = sum(abs2, x)
    penalty() = nn_params.λ_regularization*sum(sqnorm, Flux.params(f))
    accuracy(x, y) = mean(sign.(f(x)) .== sign.(y))
    loss(x, y) = nn_params.loss_func(f(x), y) + penalty()

    # TODO: Include action/policy vector and change loss to include CE-loss

    opt = ADAM(nn_params.learning_rate)
    θ = Flux.params(f)

    training_epochs = nn_params.training_epochs
    losses_train = []
    losses_valid = []
    accs_train = []
    accs_valid = []
    verbose && @info "Beginning training $(size(x_train))"
    for e in 1:training_epochs
        for (x, y) in train_data
            _, back = Flux.pullback(() -> loss(x, y), θ)
            Flux.update!(opt, θ, back(1.0f0))
        end
        loss_train = loss(x_train, y_train)
        loss_valid = loss(x_valid, y_valid)
        acc_train = accuracy(x_train, y_train)
        acc_valid = accuracy(x_valid, y_valid)
        push!(losses_train, loss_train)
        push!(losses_valid, loss_valid)
        push!(accs_train, acc_train)
        push!(accs_valid, acc_valid)
        if verbose && e % nn_params.verbose_update_frequency == 0
            println("Epoch: ", e, " Loss Train: ", loss_train, " Loss Val: ", loss_valid, " | Acc. Train: ", acc_train, " Acc. Val: ", acc_valid)
        end
        if e % nn_params.verbose_plot_frequency == 0
            plot(xlims=(1, training_epochs), ylims=(0, nn_params.normalize_target ? 1 : 2000)) # TODO: Generalize
            plot!(1:e, losses_train, label="training")
            plot!(1:e, losses_valid, label="validation")
            display(plot!())
        end
    end

    if nn_params.verbose_plot_frequency != Inf
        learning_curve = plot!()
        display(learning_curve)

        value_model = (cpu(f(x_valid))' .* std_y) .+ mean_y
        value_data = (cpu(y_valid)' .* std_y) .+ mean_y
        value_distribution = histogram(value_model, alpha=0.5, label="model", c=3)
        histogram!(value_data, alpha=0.5, label="data", c=4)
        display(value_distribution)
    end

    # Place network on the CPU (better GPU memory conservation when doing parallelized inference)
    f = cpu(f)

    # Clean GPU memory explicitly
    if device == gpu
        x_train = y_train = x_valid = y_valid = nothing
        GC.gc()
        Flux.CUDA.reclaim()
    end

    # Add un-normalization layer
    unnormalize = y -> (y .* std_y) .+ mean_y
    f = Chain(f.layers..., unnormalize)

    return f
end


"""
Evaluate the neural network `f` using the `belief` as input.
Note, inference is done on the CPU given a single input.
"""
function value_lookup(nn_params, belief, f)
    b = input_representation(belief)
    b = Float32.(b)
    x = Flux.unsqueeze(b; dims=ndims(b)+1)
    y = f(x) # evaluate network `f`
    value = cpu(y)[1] # returns 1 element 1D array
    return value
end


"""
Compare previous and current neural networks using MCTS simulations.
Use upper confidence bound on the discounted return as the comparison metric.
"""
function evaluate_agent(pomdp::POMDP, solver::BetaZeroSolver, f_prev, f_curr; outer_iter=0)
    # Run a number of simulations to evaluate the two neural networks using MCTS (`f_prev` and `f_curr`)
    solver.verbose && @info "Evaluting networks..."
    returns_prev = generate_data(pomdp, solver, f_prev; inner_iter=solver.n_evaluate, outer_iter=outer_iter)[:G]
    returns_curr = generate_data(pomdp, solver, f_curr; inner_iter=solver.n_evaluate, outer_iter=outer_iter)[:G]

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


"""
Generate training data using online MCTS with the best network so far `f` (parallelized across episodes).
"""
function generate_data(pomdp::POMDP, solver::BetaZeroSolver, f; outer_iter::Int=0, inner_iter::Int=solver.n_data_gen, store_metrics::Bool=false)
    # Confirm that network is on the CPU for inference
    f = cpu(f)

    # Run MCTS to generate data using the neural network `f`
    isnothing(solver.bmdp) && fill_bmdp!(pomdp, solver)
    solver.mcts_solver.estimate_value = (bmdp,b,d)->value_lookup(solver.network_params, b, f)
    mcts_planner = solve(solver.mcts_solver, solver.bmdp)
    up = solver.updater
    ds0 = POMDPs.initialstate_distribution(pomdp)

    # (nprocs() < nbatches) && addprocs(nbatches - nprocs())
    # @info "Number of processes: $(nprocs())"

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
            data, metrics = run_simulation(pomdp, solver, mcts_planner, up, b0, s0)
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
    return (X=X, Y=Y, G=G)
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
function run_simulation(pomdp::POMDP, solver::BetaZeroSolver, policy::POMDPs.Policy, up::POMDPs.Updater, b0, s0; max_steps=100)
    rewards::Vector{Float64} = [0.0]
    data = [BetaZeroTrainingData(b=input_representation(b0))]
    local action
    local T
    for (a,r,bp,t) in stepthrough(pomdp, policy, up, b0, s0, "a,r,bp,t", max_steps=max_steps)
        # @info "Simulation time step $t"
        T = t
        action = a
        push!(rewards, r)
        push!(data, BetaZeroTrainingData(b=input_representation(bp)))
    end

    γ = POMDPs.discount(pomdp)
    G = compute_returns(rewards; γ=γ)

    for (t,d) in enumerate(data)
        d.z = G[t]
    end

    metrics = compute_performance_metrics(pomdp, solver, data, b0, s0, action, T)

    return data, metrics
end


"""
Method to collect performance and validation metrics during BetaZero policy iteration.
Note, user defines `solver.accuracy_func` to determine the accuracy of the final decision (if applicable).
"""
function compute_performance_metrics(pomdp::POMDP, solver::BetaZeroSolver, data, b0, s0, action, T)
    # - mean discounted return over time
    # - accuracy over time (i.e., did it make the correct decision, if there's some notion of correct)
    # - number of actions (e.g., number of drills for mineral exploration)
    metrics = nothing
    if solver.collect_metrics
        returns = [d.z for d in data]
        discounted_return = returns[1]
        accuracy = solver.accuracy_func(pomdp, b0, s0, action, returns) # NOTE: Problem specific, provide function to compute this
        metrics = (discounted_return=discounted_return, accuracy=accuracy, num_actions=T)
    end
    return metrics
end


"""
Run a test on a holdout set to collect performance metrics during BetaZero policy iteration.
"""
function run_holdout_test!(pomdp::POMDP, solver::BetaZeroSolver, f; outer_iter::Int=0)
    if solver.n_holdout > 0
        solver.verbose && @info "Running holdout test..."
        returns = generate_data(pomdp, solver, f; inner_iter=solver.n_holdout, outer_iter=outer_iter, store_metrics=true)[:G]
        solver.verbose && display(UnicodePlots.histogram(returns))
        solver.verbose && @show mean_and_std(returns)
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
Get action from BetaZero policy (online MCTS using value & policy network).
"""
function POMDPs.action(policy::BetaZeroPolicy, b)
    return action(policy.planner, b)
end


end # module
