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
    verbose_update_frequency::Int = 1 # Frequency of printed training output
    verbose_plot_frequency::Number = 10 # Frequency of plotted training/validation output
end


@with_kw mutable struct BetaZeroSolver <: POMDPs.Solver
    n_iterations::Int = 100
    n_data_gen::Int = 100
    n_evaluate::Int = 5 # TODO: Change to 100
    updater::POMDPs.Updater
    network_params::BetaZeroNetworkParameters = BetaZeroNetworkParameters() # parameters for training CNN
    belief_reward::Function = (pomdp::POMDP, b, a, bp)->0.0
    # belief representation function (see `representation.jl` TODO: should it be a parameter or overloaded function?)
    mcts_solver::AbstractMCTSSolver = DPWSolver(n_iterations=10,
                                                check_repeat_action=true,
                                                exploration_constant=100.0,
                                                k_action=2.0,
                                                alpha_action=0.25,
                                                tree_in_info=true,
                                                show_progress=false,
                                                estimate_value=(bmdp,b,d)->0.0) # `estimate_value` will be replaced with a neural network lookup
    bmdp::Union{BeliefMDP,Nothing} = nothing # Belief-MDP version of the POMDP
    collect_metrics::Bool = false # Indicate that performance metrics should be collected.
    performance_metrics::Array = []
    accuracy_func::Function = (pomdp,belief,state,action,returns)->nothing # (returns Bool): Function to indicate that the decision was "correct" (if applicable)
end


@with_kw mutable struct BetaZeroTrainingData
    b = nothing # current belief
    π = nothing # current policy estimate (using N(s,a))
    z = nothing # final discounted return of the episode
end


"""
The main BetaZero policy iteration algorithm.
"""
function POMDPs.solve(solver::BetaZeroSolver, pomdp::POMDP)
    solver.bmdp = BeliefMDP(pomdp, solver.updater, solver.belief_reward)
    f_prev = initialize_network(solver)

    @time for i in 1:solver.n_iterations
        @info "BetaZero iteration $i/$(solver.n_iterations)"

        # 1) Generate data using the best BetaZero agent so far.
        data = generate_data(pomdp, solver, f_prev; iter=i)

        # 2) Optimize neural network parameters with recent simulated data.
        f_curr = train_network(f_prev, data, solver.network_params)

        # 3) BetaZero agent is evaluated (compared to previous agent, beating it in in returns).
        f_prev = f_curr
        # f_prev = evaluate_agent(f_prev, f_curr) # TODO.
    end

    solver.mcts_solver.estimate_value = (bmdp,b,d)->value_lookup(solver.network_params, b, f_prev)
    mcts_planner = solve(solver.mcts_solver, solver.bmdp)
    policy = BetaZeroPolicy(f_prev, mcts_planner)

    return policy
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

    f = Chain(
        Conv(filter, input_size[end]=>num_filters1, relu),
        Conv(filter, num_filters1=>num_filters2, relu),
        Flux.flatten,
        Dense(out_conv_size, num_dense1, relu),
        Dense(num_dense1, num_dense2, relu),
        Dense(num_dense2, out_dim),
    )

    return nn_params.device(f)
end


"""
Train policy & value neural network `f` using the latest `data` generated from online tree search (MCTS).
"""
function train_network(f, data, nn_params::BetaZeroNetworkParameters)
    x_data, y_data = data.X, data.Y

    if nn_params.normalize_target
        mean_y = mean(y_data)
        std_y = std(y_data)
        y_data = (y_data .- mean_y) ./ std_y
    end

    n_data = length(y_data)
    n_train = Int(n_data ÷ (1/nn_params.training_split))

    @info "Data set size: $n_data"

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

    train_data = Flux.Data.DataLoader((x_train, y_train), batchsize=nn_params.batchsize, shuffle=true)

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
    @info "Beginning training $(size(x_train))"
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
        if e % nn_params.verbose_update_frequency == 0
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

    # Clean GPU memory explicitly
    if device == gpu
        x_train = y_train = x_valid = y_valid = nothing
        GC.gc()
        Flux.CUDA.reclaim()
    end

    return f
end


"""
Evaluate the neural network `f` using the `belief` as input.
"""
function value_lookup(nn_params, belief, f)
    device = nn_params.device
    b = input_representation(belief)
    b = Float32.(b)
    x = Flux.unsqueeze(b; dims=ndims(b)+1)
    x = device(x) # put data on CPU or GPU (i.e., `device`)
    y = f(x) # evaluate network `f`
    value = cpu(y)[1] # returns 1 element 1D array
    return value
end


function evaluate_agent(f_prev, f_curr; simulations=100)
    prev_correct = 0
    curr_correct = 0

    # Run a number of simulations to evaluate the two neural networks using MCTS (`f_prev` and `f_curr`)
    for i in 1:simulations
        ans_prev = evaluate(f_prev; n_evaluate) # TODO.
        ans_curr = evaluate(f_curr; n_evaluate) # TODO.
        ans_true = truth() # TODO.
        if ans_prev == ans_true
            prev_correct += 1
        end
        if ans_curr == ans_true
            curr_correct += 1
        end
    end

    if curr_correct >= prev_correct
        return f_curr
    else
        return f_prev
    end
end


"""
Generate training data using online MCTS with the best network so far `f` (parallelized across episodes).
"""
function generate_data(pomdp::POMDP, solver::BetaZeroSolver, f; iter::Int=0)
    # Run MCTS to generate data using the neural network `f`
    solver.mcts_solver.estimate_value = (bmdp,b,d)->value_lookup(solver.network_params, b, f)
    mcts_planner = solve(solver.mcts_solver, solver.bmdp)
    up = solver.updater
    ds0 = POMDPs.initialstate_distribution(pomdp)

    # (nprocs() < nbatches) && addprocs(nbatches - nprocs())
    @info "Number of processes: $(nprocs())"

    progress = Progress(solver.n_data_gen)
    channel = RemoteChannel(()->Channel{Bool}(), 1)

    @async while take!(channel)
        next!(progress)
    end

    @time parallel_data = pmap(i->begin
            # TODO: Batches!
            seed = parse(Int, string(iter, lpad(i, length(digits(solver.n_data_gen)), '0'))) # 1001, 1002, etc. for BetaZero iter=1
            Random.seed!(seed)
            @info "Generating data ($i/$(solver.n_data_gen)) with seed ($seed)"
            s0 = rand(ds0)
            b0 = POMDPs.initialize_belief(up, ds0)
            data, metrics = run_simulation(pomdp, solver, mcts_planner, up, b0, s0)
            B = []
            Z = []
            # Π = []
            for d in data
                push!(B, d.b)
                push!(Z, d.z)
                # push!(Π, d.π) # TODO.
            end
            put!(channel, true) # trigger progress bar update
            B, Z, metrics
        end, 1:solver.n_data_gen)

    put!(channel, false) # tell printing task to finish

    beliefs = vcat([d[1] for d in parallel_data]...) # combine all beliefs
    returns = vcat([d[2] for d in parallel_data]...) # combine all returns
    metrics = vcat([d[3] for d in parallel_data]...) # combine all metrics

    push!(solver.performance_metrics, metrics...)

    ndims_belief = ndims(beliefs[1])
    X = cat(beliefs...; dims=ndims_belief+1)
    Y = reshape(Float32.(returns), 1, length(returns))
    return (X=X, Y=Y)
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
    rewards = [0.0]
    data = [BetaZeroTrainingData(b=input_representation(b0))]
    local action
    local T
    for (a,r,bp,t) in stepthrough(pomdp, policy, up, b0, s0, "a,r,bp,t", max_steps=max_steps)
        @info "Simulation time step $t"
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
