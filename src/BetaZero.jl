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
    BetaZeroPolicy

# function policy_iteration()
#     policy_evaluation()
#     policy_improvement()
# end

# function policy_evaluation() end
# function policy_improvement() end


mutable struct BetaZeroPolicy <: POMDPs.Policy
    network::Chain
    mcts_planner::AbstractMCTSPlanner
end


@with_kw mutable struct BetaZeroNetworkParameters
    input_size = (30,30,5)
    training_epochs = 300 # TODO: Change to 1000.
    normalize_target::Bool = true # Normalize target data to standard normal (0 mean)
    training_split::Float64 = 0.8 # Training / validation split (Default: 80/20)
    batchsize::Int = 512
    learning_rate::Float64 = 0.01 # Learning rate for ADAM optimizer during training
    λ_regularization::Float64 = 0.0001 # Parameter for L2-norm regularization
    loss_func::Function = Flux.Losses.mae # MAE works well for problems with large returns around zero, and spread out otherwise.
    device = gpu
    verbose_update_frequency::Int = 1 # Frequency of printed training output
    verbose_plot_frequency::Int = 10 # Frequency of plotted training/validation output
end


@with_kw mutable struct BetaZeroSolver <: POMDPs.Solver
    n_iterations::Int = 100
    n_data_gen::Int = 10 # TODO: Change to ~ 100-1000
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
    f_prev = f = initialize_network(solver)
    data = generate_data(pomdp, solver, f_prev)
    # policy = BetaZeroPolicy() # TODO.

    for i in 1:1 # TODO: solver.interations
        # 1) Optimize neural network parameters with recent simulated data.
        f_curr = train_network(f, data, solver.network_params)

        # 2) BetaZero agent is evaluated (compared to previous agent, beating it in 55%+ simulations).
        f = f_curr
        # f = evaluate_agent(f_prev, f_curr)

        # 3) Generate new data using the best BetaZero agent so far.
        # data = generate_data(pomdp, f)
    end

    # return policy
    # return f
    return f, data
end


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
    )
end


function train_network(f, data, nn_params::BetaZeroNetworkParameters)
    x_data, y_data = data.X, data.Y

    if nn_params.normalize_target
        mean_y = mean(y_data)
        std_y = std(y_data)
        y_data = (y_data .- mean_y) ./ std_y
    end

    n_data = length(y_data)
    n_train = Int(n_data ÷ (1/nn_params.training_split))

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

    learning_curve = plot!()
    display(learning_curve)

    value_model = (cpu(f(x_valid))' .* std_y) .+ mean_y
    value_data = (cpu(y_valid)' .* std_y) .+ mean_y
    value_distribution = histogram(value_model, alpha=0.5, label="model", c=3)
    histogram!(value_data, alpha=0.5, label="data", c=4)
    display(value_distribution)

    return f
end


function value_lookup(belief, f)
    b = input_representation(belief)
    b = Float32.(b)
    x = Flux.unsqueeze(b; dims=ndims(b)+1)
    value = first(f(x)) # returns 1 element 1D array
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


function generate_data(pomdp::POMDP, solver::BetaZeroSolver, f)
    # Run MCTS to generate data using the neural network `f`
    solver.mcts_solver.estimate_value = (bmdp,b,d)->value_lookup(b,f)
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
            @info "Generating data ($i/$(solver.n_data_gen))"
            s0 = rand(ds0)
            b0 = POMDPs.initialize_belief(up, ds0)
            data = run_simulation(pomdp, mcts_planner, up, b0, s0)
            B = []
            Z = []
            # Π = []
            for d in data
                push!(B, d.b)
                push!(Z, d.z)
                # push!(Π, d.π) # TODO.
            end
            put!(channel, true) # trigger progress bar update
            B, Z
        end, 1:solver.n_data_gen)

    put!(channel, false) # tell printing task to finish

    beliefs = vcat(first.(parallel_data)...) # combine all beliefs
    returns = vcat(last.(parallel_data)...) # combine all returns

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
function run_simulation(pomdp::POMDP, policy::POMDPs.Policy, up::POMDPs.Updater, b0, s0; max_steps=100)
    rewards = [0.0]
    data = [BetaZeroTrainingData(b=input_representation(b0))]
    for (r,bp,t) in stepthrough(pomdp, policy, up, b0, s0, "r,bp,t", max_steps=max_steps)
        @info "Simulation time step $t"
        push!(rewards, r)
        push!(data, BetaZeroTrainingData(b=input_representation(bp)))
    end

    γ = POMDPs.discount(pomdp)
    G = compute_returns(rewards; γ=γ)

    s_massive = s0.ore_map .>= pomdp.massive_threshold
    massive = pomdp.dim_scale*sum(s_massive)

    @show massive
    @show G
    @show rewards

    for (t,d) in enumerate(data)
        d.z = G[t]
    end

    return data
end


function collect_metrics()
    # TODO: Collect intermediate results from the steps above.
end


end # module
