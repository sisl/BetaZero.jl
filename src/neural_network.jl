"""
Initialize policy & value network with random weights.
"""
initialize_network(solver::BetaZeroSolver) = initialize_network(solver.nn_params)
function initialize_network(nn_params::BetaZeroNetworkParameters) # LeNet5
    input_size = nn_params.input_size
    action_size = nn_params.action_size
    activation = nn_params.activation
    ℓs = nn_params.layer_size

    use_dropout = nn_params.use_dropout
    p_dropout = nn_params.p_dropout
    use_batchnorm = nn_params.use_batchnorm
    batchnorm_momentum = nn_params.batchnorm_momentum

    function DenseLayer(in_out::Pair)
        input, output = in_out
        if use_batchnorm && !use_dropout
            return [Dense(input => output), BatchNorm(output, activation, momentum=batchnorm_momentum)]
        elseif use_dropout && !use_batchnorm
            return [Dense(input => output, activation), Dropout(p_dropout)]
        elseif use_batchnorm && use_dropout
            return [Dense(input => output), BatchNorm(output, activation, momentum=batchnorm_momentum), Dropout(p_dropout)]
        else
            return [Dense(input => output, activation)]
        end
    end

    if nn_params.use_cnn
        cnn_params = nn_params.cnn_params
        filter = cnn_params.filter
        num_filters1 = cnn_params.num_filters[1]
        num_filters2 = cnn_params.num_filters[2]
        out_conv_size = prod([input_size[1] - 2*(filter[1]-1), input_size[2] - 2*(filter[2]-1), num_filters2])
        num_dense1 = cnn_params.num_dense[1]
        num_dense2 = cnn_params.num_dense[2]

        return Chain(
            Conv(filter, input_size[end]=>num_filters1, activation),
            Conv(filter, num_filters1=>num_filters2, activation),
            Flux.flatten,
            DenseLayer(out_conv_size=>num_dense1)...,
            DenseLayer(num_dense1=>num_dense2)...,
            Parallel(vcat,
                value_head = Chain(
                    DenseLayer(num_dense2 => ℓs)...,
                    Dense(ℓs => 1),
                    # Note: A normalization layer will be added during training (with the old layer removed before the next training phase).
                ),
                policy_head = Chain(
                    DenseLayer(num_dense2 => ℓs)...,
                    Dense(ℓs => action_size),
                    softmax,
                )
            )
        )
    else
        return Chain(
            DenseLayer(prod(input_size) => ℓs)...,
            DenseLayer(ℓs => ℓs)...,
            DenseLayer(ℓs => ℓs)...,
            Parallel(vcat,
            value_head = Chain(
                    DenseLayer(ℓs => ℓs)...,
                    Dense(ℓs => 1),
                    # Note: A normalization layer will be added during training (with the old layer removed before the next training phase).
                ),
                policy_head = Chain(
                    DenseLayer(ℓs => ℓs)...,
                    Dense(ℓs => action_size),
                    softmax,
                )
            )
        )
    end
end


"""
Train policy & value neural network `f` using the latest `data` generated from online tree search (MCTS).
"""
function train(f::Chain, solver::BetaZeroSolver; verbose::Bool=false, results=nothing)
    nn_params = solver.nn_params
    lr = nn_params.learning_rate
    λ = nn_params.λ_regularization
    loss_str = string(nn_params.loss_func)
    normalize_input = nn_params.normalize_input
    normalize_output = nn_params.normalize_output
    key = (lr, λ, loss_str, normalize_input, normalize_output)

    n_train = Int(nn_params.n_samples ÷ (1/nn_params.training_split))
    n_valid = nn_params.n_samples - n_train

    data_train = sample_data(solver.data_buffer_train, n_train) # sample from last `n_buffer` simulations.
    data_valid = sample_data(solver.data_buffer_valid, n_valid) # sample from last `n_buffer` simulations.
    x_train, y_train = data_train.X, data_train.Y
    x_valid, y_valid = data_valid.X, data_valid.Y

    normalize_func(x, μ, σ) = (x .- μ) ./ σ

    # Normalize input values close to the range of [-1, 1]
    if normalize_input
        # Normalize only based on the training data (but apply it to training and validation data)
        mean_x = mean(x_train, dims=ndims(x_train))
        std_x = std(x_train, dims=ndims(x_train))
        x_train = normalize_func(x_train, mean_x, std_x)
        x_valid = normalize_func(x_valid, mean_x, std_x)
    end

    # Normalize target values close to the range of [-1, 1]
    if normalize_output
        # Normalize only based on the training data (but apply it to training and validation data)
        mean_y = mean(y_train[1,:])
        std_y = std(y_train[1,:])
        y_train[1,:] = normalize_func(y_train[1,:], mean_y, std_y)
        y_valid[1,:] = normalize_func(y_valid[1,:], mean_y, std_y)
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
    heads = f.layers[end]
    value_head = heads.layers.value_head
    if isa(value_head.layers[end], Function) && normalize_output
        policy_head = heads.layers.policy_head
        value_head = Chain(value_head.layers[1:end-1]...)
        heads = Parallel(heads.connection, value_head=value_head, policy_head=policy_head)
        f = Chain(f.layers[1:end-1]..., heads)
    end

    # Put network on GPU for training
    f = device(f)

    sqnorm(x) = sum(abs2, x)
    penalty() = λ*sum(sqnorm, Flux.params(f))
    sign_accuracy(x, y) = mean(sign.(f(x)[1,:]) .== sign.(y[1,:]))
    loss(x, y) = begin
        local ỹ = f(x)
        # vmask = Flux.CuArray([1,0,0,0])
        vmask = Flux.CuArray(vcat(1, zeros(Int,size(ỹ,1)-1)))
        pmask = 1 .- vmask
        v = vmask .* ỹ # value prediction
        𝐩 = pmask .* ỹ # policy prediction
        z = vmask .* y # true value
        π = pmask .* y # true policy vector
        nn_params.loss_func(v, z) + Flux.Losses.crossentropy(𝐩, π) + penalty()
    end

    opt = nn_params.optimizer(lr)
    θ = Flux.params(f)

    training_epochs = nn_params.training_epochs
    losses_train = []
    losses_valid = []
    accs_train = []
    accs_valid = []
    learning_curve = nothing
    checkpoint_loss_valid = Inf
    checkpoint_loss_train = Inf
    checkpoint_acc_valid = 0
    checkpoint_acc_train = 0
    f_checkpoint = f

    local loss_train = Inf
    local loss_valid = Inf
    local acc_train = 0
    local acc_valid = 0
    local checkpoint_epoch = Inf

    logging_fn(epoch, loss_train, loss_valid, acc_train, acc_valid; extra="", digits=5) = string("Epoch: ", epoch, "\t Loss Train: ", round(loss_train; digits), "\t Loss Val: ", round(loss_valid; digits), " \t|\t Sign Acc. Train: ", rpad(round(acc_train; digits), digits+2, '0'), "\t Sign Acc. Val: ", rpad(round(acc_valid; digits), digits+2, '0'), extra)

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
            println(logging_fn(e, loss_train, loss_valid, acc_train, acc_valid))
        end
        if e % nn_params.checkpoint_frequency == 0
            checkpoint_condition = nn_params.checkpoint_validation_loss ? loss_valid < checkpoint_loss_valid : loss_train < checkpoint_loss_train
            if checkpoint_condition
                checkpoint_loss_valid = loss_valid
                checkpoint_loss_train = loss_train
                checkpoint_acc_valid = acc_valid
                checkpoint_acc_train = acc_train
                checkpoint_epoch = e
                f_checkpoint = deepcopy(f)
                verbose && println(logging_fn(e, loss_train, loss_valid, acc_train, acc_valid; extra=" [Checkpoint]"))
            end
        end

        if e % nn_params.verbose_plot_frequency == 0
            # TODO: Generalize y-values
            learning_curve = plot(xlims=(1, training_epochs), ylims=(0, 1), title="learning curve: $key")
            plot!(1:e, losses_train, label="training")
            plot!(1:e, losses_valid, label="validation")
            display(learning_curve)
        end
    end

    if nn_params.use_checkpoint
        # check final loss
        checkpoint_condition = nn_params.checkpoint_validation_loss ? loss_valid < checkpoint_loss_valid : loss_train < checkpoint_loss_train
        if checkpoint_condition
            checkpoint_loss_valid = loss_valid
            checkpoint_loss_train = loss_train
            checkpoint_acc_valid = acc_valid
            checkpoint_acc_train = acc_train
            checkpoint_epoch = training_epochs
            f_checkpoint = deepcopy(f)
        end
        verbose && println(logging_fn(checkpoint_epoch, checkpoint_loss_train, checkpoint_loss_valid, checkpoint_acc_train, checkpoint_acc_valid; extra=" [Final network checkpoint]"))
        f = f_checkpoint
    end

    # Place network on the CPU (better GPU memory conservation when doing parallelized inference)
    f = cpu(f)

    if normalize_input
        # Add normalization input layer
        unnormalize_x = x -> (x .* std_x) .+ mean_x
        normalize_x = x -> (x .- mean_x) ./ std_x
        f = Chain(normalize_x, f.layers...)
    end

    if normalize_output
        # Add un-normalization output layer
        unnormalize_y = y -> (y .* std_y) .+ mean_y
        heads = f.layers[end]
        value_head = heads.layers.value_head
        value_head = Chain(value_head.layers..., unnormalize_y)
        policy_head = heads.layers.policy_head
        heads = Parallel(heads.connection, value_head=value_head, policy_head=policy_head)
        f = Chain(f.layers[1:end-1]..., heads)
    end

    value_model = normalize_input ? f(unnormalize_x(cpu(x_valid)))[1,:] : f(cpu(x_valid))[1,:]
    value_data = normalize_output ? unnormalize_y(cpu(y_valid))[1,:] : cpu(y_valid)[1,:]

    if nn_params.verbose_plot_frequency != Inf
        value_distribution = nothing
        try
            value_distribution = Plots.histogram(value_model, alpha=0.5, label="model", c=:gray, title="values: $key")
            Plots.histogram!(value_data, alpha=0.5, label="data", c=:navy)
            display(value_distribution)
            plot_bias(value_model, value_data)
            display(Plots.title!("validation data"))

            value_model_training = normalize_input ? f(unnormalize_x(cpu(x_train)))[1,:] : f(cpu(x_train))[1,:]
            value_data_training = normalize_output ? unnormalize_y(cpu(y_train))[1,:] : cpu(y_train)[1,:]
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

        results[key]["data"] = (train=(X=x_train, Y=y_train), valid=(X=x_valid, Y=y_valid))
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
Evaluate the neural network `f` using the `belief` as input, return the predicted value.
Note, inference is done on the CPU given a single input.
"""
function value_lookup(belief, f::Union{Chain,EnsembleNetwork})
    b = Float32.(input_representation(belief))
    x = Flux.unsqueeze(b; dims=ndims(b)+1) # add extra single dimension (batch)
    y = f(x) # evaluate network `f`
    value = cpu(y)
    return value[1]
    # return length(value) == 1 ? value[1] : value
end


"""
Evaluate the neural network `f` using the `belief` as input, return the predicted policy vector.
Note, inference is done on the CPU given a single input.
"""
function policy_lookup(belief, f::Union{Chain,EnsembleNetwork})
    b = Float32.(input_representation(belief))
    x = Flux.unsqueeze(b; dims=ndims(b)+1) # add extra single dimension (batch)
    y = f(x) # evaluate network `f`
    policy = cpu(y)
    return policy[2:end]
end


"""
Use predicted policy vector to sample next action.
"""
function next_action(problem::Union{BeliefMDP, POMDP}, belief, f::Union{Chain,EnsembleNetwork})
    p = policy_lookup(belief, f)
    as = POMDPs.actions(problem) # TODO: actions(problem, belief) ??
    return rand(SparseCat(as, p))
end


"""
Sweep neural network hyperparameters to tune.
"""
function tune_network_parameters(pomdp::POMDP, solver::BetaZeroSolver;
                                 learning_rates=[0.1, 0.01, 0.005, 0.001, 0.0001],
                                 λs=[0, 0.1, 0.005, 0.001, 0.0001],
                                 loss_funcs=[Flux.Losses.mae, Flux.Losses.mse],
                                 normalize_outputs=[true, false])
    _use_random_policy_data_gen = solver.use_random_policy_data_gen # save original setting
    solver.use_random_policy_data_gen = true
    @info "Tuning using a random policy for data generation."
    results = Dict()
    N = sum(map(length, [learning_rates, λs, loss_funcs, normalize_outputs]))
    i = 1
    for normalize_output in normalize_outputs
        for loss in loss_funcs
            for λ in λs
                for lr in learning_rates
                    @info "Tuning iteration: $i/$N ($(round(i/N*100, digits=3)))"
                    i += 1

                    solver.nn_params.learning_rate = lr
                    solver.nn_params.λ_regularization = λ
                    solver.nn_params.loss_func = loss
                    solver.nn_params.normalize_output = normalize_output
                    loss_str = string(loss)

                    @info "Tuning with: lr=$lr, λ=$λ, loss=$loss_str, normalize_output=$normalize_output"
                    empty!(solver.data_buffer_train)
                    empty!(solver.data_buffer_valid)
                    f_prev = initialize_network(solver)
                    generate_data!(pomdp, solver, f_prev; use_random_policy=solver.use_random_policy_data_gen, inner_iter=solver.n_data_gen, outer_iter=1)
                    f_curr = train(deepcopy(f_prev), solver; verbose=solver.verbose, results=results)

                    key = (lr, λ, loss_str, normalize_output)
                    results[key]["network"] = f_curr
                end
            end
        end
    end
    solver.use_random_policy_data_gen = _use_random_policy_data_gen # reset to original setting
    return results
end
