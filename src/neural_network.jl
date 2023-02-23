"""
Initialize policy & value network with random weights.
"""
initialize_network(solver::BetaZeroSolver) = initialize_network(solver.nn_params)
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
function train(f::Chain, solver::BetaZeroSolver; verbose::Bool=false, results=nothing)
    nn_params = solver.nn_params
    lr = nn_params.learning_rate
    λ = nn_params.λ_regularization
    loss_str = string(nn_params.loss_func)
    normalize_target = nn_params.normalize_target
    key = (lr, λ, loss_str, normalize_target)

    n_train = Int(nn_params.n_samples ÷ (1/nn_params.training_split))
    n_valid = nn_params.n_samples - n_train

    data_train = sample_data(solver.data_buffer_train, n_train) # sample from last `n_buffer` simulations.
    data_valid = sample_data(solver.data_buffer_valid, n_valid) # sample from last `n_buffer` simulations.
    x_train, y_train = data_train.X, data_train.Y
    x_valid, y_valid = data_valid.X, data_valid.Y

    normalize_func(x, μ, σ) = (x .- μ) ./ σ
    
    # Normalize input values close to the range of [-1, 1]
    normalize_input = true
    if normalize_input
        # Normalize only based on the training data (but apply it to training and validation data)
        mean_x = mean(x_train, dims=ndims(x_train))
        std_x = std(x_train, dims=ndims(x_train))
        x_train = normalize_func(x_train, mean_x, std_x)
        x_valid = normalize_func(x_valid, mean_x, std_x)
    end

    # Normalize target values close to the range of [-1, 1]
    if normalize_target
        # Normalize only based on the training data (but apply it to training and validation data)
        mean_y = mean(y_train)
        std_y = std(y_train)
        y_train = normalize_func(y_train, mean_y, std_y)
        y_valid = normalize_func(y_valid, mean_y, std_y)
    end

    verbose && @info "Data set size: $(n_train):$(n_valid) (training:validation)"

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
    if isa(f.layers[1], Function) && normalize_input
        f = Chain(f.layers[2:end]...)
    end

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
    learning_curve = nothing

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
            learning_curve = plot(xlims=(1, training_epochs), ylims=(0, 1), title="learning curve: $key")
            plot!(1:e, losses_train, label="training")
            plot!(1:e, losses_valid, label="validation")
            display(learning_curve)
        end
    end

    # Place network on the CPU (better GPU memory conservation when doing parallelized inference)
    f = cpu(f)

    if normalize_input
        # Add normalization input layer
        unnormalize_x = x -> (x .* std_x) .+ mean_x
        normalize_x = x -> (x .- mean_x) ./ std_x
        f = Chain(normalize_x, f.layers...)
    end

    if normalize_target
        # Add un-normalization output layer
        unnormalize_y = y -> (y .* std_y) .+ mean_y
        f = Chain(f.layers..., unnormalize_y)
    end

    value_model = normalize_input ? f(unnormalize_x(cpu(x_valid)))' : f(cpu(x_valid))'
    value_data = normalize_target ? unnormalize_y(cpu(y_valid))' : cpu(y_valid)'

    if nn_params.verbose_plot_frequency != Inf
        value_distribution = nothing
        try
            value_distribution = Plots.histogram(value_model, alpha=0.5, label="model", c=:gray, title="values: $key")
            Plots.histogram!(value_data, alpha=0.5, label="data", c=:navy)
            display(value_distribution)
            plot_bias(value_model, value_data)
            display(Plots.title!("validation data"))

            value_model_training = normalize_input ? f(unnormalize_x(cpu(x_train)))' : f(cpu(x_train))'
            value_data_training = normalize_target ? unnormalize_y(cpu(y_train))' : cpu(y_train)'
            plot_bias(value_model_training, value_data_training)
            display(Plots.title!("training data"))
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
Note, inference is done on the CPU given a single input.
"""
function value_lookup(belief, f::Chain)
    b = Float32.(input_representation(belief))
    x = Flux.unsqueeze(b; dims=ndims(b)+1) # add extra single dimension (batch)
    y = f(x) # evaluate network `f`
    value = cpu(y)
    return length(value) == 1 ? value[1] : value
end


"""
Sweep neural network hyperparameters to tune.
"""
function tune_network_parameters(pomdp::POMDP, solver::BetaZeroSolver;
                                 learning_rates=[0.1, 0.01, 0.005, 0.001, 0.0001],
                                 λs=[0, 0.1, 0.005, 0.001, 0.0001],
                                 loss_funcs=[Flux.Losses.mae, Flux.Losses.mse],
                                 normalize_targets=[true, false])
    _use_random_policy_data_gen = solver.use_random_policy_data_gen # save original setting
    solver.use_random_policy_data_gen = true
    @info "Tuning using a random policy for data generation."
    results = Dict()
    N = sum(map(length, [learning_rates, λs, loss_funcs, normalize_targets]))
    i = 1
    for normalize_target in normalize_targets
        for loss in loss_funcs
            for λ in λs
                for lr in learning_rates
                    @info "Tuning iteration: $i/$N ($(round(i/N*100, digits=3)))"
                    i += 1

                    solver.nn_params.learning_rate = lr
                    solver.nn_params.λ_regularization = λ
                    solver.nn_params.loss_func = loss
                    solver.nn_params.normalize_target = normalize_target
                    loss_str = string(loss)

                    @info "Tuning with: lr=$lr, λ=$λ, loss=$loss_str, normalize_target=$normalize_target"
                    empty!(solver.data_buffer_train)
                    empty!(solver.data_buffer_valid)
                    f_prev = initialize_network(solver)
                    generate_data!(pomdp, solver, f_prev; use_random_policy=solver.use_random_policy_data_gen, inner_iter=solver.n_data_gen, outer_iter=1)
                    f_curr = train(deepcopy(f_prev), solver; verbose=solver.verbose, results=results)

                    key = (lr, λ, loss_str, normalize_target)
                    results[key]["network"] = f_curr
                end
            end
        end
    end
    solver.use_random_policy_data_gen = _use_random_policy_data_gen # reset to original setting
    return results
end
