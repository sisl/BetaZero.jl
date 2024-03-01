"""
Initialize policy & value network with random weights.
"""
initialize_network(solver::BetaZeroSolver) = initialize_network(solver.nn_params)
function initialize_network(nn_params::BetaZeroNetworkParameters)
    input_size = nn_params.input_size
    action_size = nn_params.action_size
    activation = nn_params.activation
    ℓs = nn_params.layer_size

    use_dropout = nn_params.use_dropout
    p_dropout = nn_params.p_dropout
    use_batchnorm = nn_params.use_batchnorm
    batchnorm_momentum = nn_params.batchnorm_momentum

    function DenseRegularizedLayer(in_out::Pair)
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

    function ConvRegularizedLayer(filter, in_out::Pair)
        input, output = in_out
        if use_batchnorm
            return [Conv(filter, input => output), BatchNorm(output, activation, momentum=batchnorm_momentum)]
        else
            return [Conv(filter, input => output, activation)]
        end
    end

    if nn_params.use_cnn
        cnn_params = nn_params.cnn_params
        filter = cnn_params.filter
        filter_policy = (1,1)
        filter_value = (1,1)
        num_filters1 = cnn_params.num_filters[1]
        num_filters2 = cnn_params.num_filters[2]
        num_filters_policy = 2
        num_filters_value = 1
        out_conv_size = prod([input_size[1] - 2*(filter[1]-1), input_size[2] - 2*(filter[2]-1), num_filters2])
        out_conv_size_policy = prod([input_size[1] - 2*(filter[1]-1) - (filter_policy[1]-1), input_size[2] - 2*(filter[2]-1), num_filters_policy])
        out_conv_size_value = prod([input_size[1] - 2*(filter[1]-1) - (filter_value[1]-1), input_size[2] - 2*(filter[2]-1), num_filters_value])
        num_dense1 = cnn_params.num_dense[1]
        num_dense2 = cnn_params.num_dense[2]

        if nn_params.use_deepmind_arch
            # Simplified non-resnet AlphaZero architecture.
            return Chain(
                ConvRegularizedLayer(filter, input_size[end]=>num_filters1)...,
                ConvRegularizedLayer(filter, num_filters1=>num_filters2)...,
                Parallel(vcat,
                    value_head = Chain(
                        ConvRegularizedLayer(filter_value, num_filters2=>num_filters_value)...,
                        Flux.flatten,
                        Dense(out_conv_size_value => ℓs, relu),
                        Dense(ℓs => 1),
                        # Note: A normalization layer will be added during training (with the old layer removed before the next training phase).
                    ),
                    policy_head = Chain(
                        ConvRegularizedLayer(filter_policy, num_filters2=>num_filters_policy)...,
                        Flux.flatten,
                        Dense(out_conv_size_policy => action_size),
                        softmax,
                    )
                )
            )
        else
            # LeNet5 inspired architecture (default).
            return Chain(
                Conv(filter, input_size[end]=>num_filters1, activation),
                Conv(filter, num_filters1=>num_filters2, activation),
                Flux.flatten,
                DenseRegularizedLayer(out_conv_size=>num_dense1)...,
                DenseRegularizedLayer(num_dense1=>num_dense2)...,
                Parallel(vcat,
                    value_head = Chain(
                        DenseRegularizedLayer(num_dense2 => ℓs)...,
                        Dense(ℓs => 1),
                        # Note: A normalization layer will be added during training (with the old layer removed before the next training phase).
                    ),
                    policy_head = Chain(
                        DenseRegularizedLayer(num_dense2 => ℓs)...,
                        Dense(ℓs => action_size),
                        softmax,
                    )
                )
            )
        end
    else
        # Simple fully-connected MLP (default for non-CNN inputs).
        return Chain(
            DenseRegularizedLayer(prod(input_size) => ℓs)...,
            DenseRegularizedLayer(ℓs => ℓs)...,
            DenseRegularizedLayer(ℓs => ℓs)...,
            Parallel(vcat,
                value_head = Chain(
                    DenseRegularizedLayer(ℓs => ℓs)...,
                    Dense(ℓs => 1),
                    # Note: A normalization layer will be added during training (with the old layer removed before the next training phase).
                ),
                policy_head = Chain(
                    DenseRegularizedLayer(ℓs => ℓs)...,
                    Dense(ℓs => action_size),
                    softmax,
                )
            )
        )
    end
end


calc_loss_weight(nn_params::BetaZeroNetworkParameters) = calc_loss_weight(nn_params.action_size)
calc_loss_weight(action_size::Int) = Float32(round(1 - 1/action_size; digits=2))


"""
Train policy & value neural network `f` using the latest `data` generated from online tree search (MCTS).
"""
function train(f::Chain, solver::BetaZeroSolver; verbose::Bool=false, results=nothing, ϵ_std::Float32=1f-10, rstats::RunningStats=RunningStats(), use_running_stats::Bool=true)
    nn_params = solver.nn_params
    device = nn_params.device
    lr = nn_params.learning_rate
    λ = nn_params.λ_regularization
    loss_str = string(nn_params.loss_func)
    normalize_input = nn_params.normalize_input
    normalize_output = nn_params.normalize_output
    sample_more_than_collected = nn_params.sample_more_than_collected
    value_loss_weight = nn_params.value_loss_weight
    use_kl_loss = nn_params.use_kl_loss
    key = (lr, λ, loss_str, normalize_input, normalize_output)

    n_train = Int(nn_params.n_samples ÷ (1/nn_params.training_split))
    n_valid = nn_params.n_samples - n_train

    data_train_set = sample_data(solver.data_buffer_train, n_train; sample_more_than_collected) # sample from last `n_buffer` simulations.
    data_valid_set = sample_data(solver.data_buffer_valid, n_valid; sample_more_than_collected) # sample from last `n_buffer` simulations.
    x_train, y_train = data_train_set.X, data_train_set.Y
    x_valid, y_valid = data_valid_set.X, data_valid_set.Y

    # Update training and validation set size (could be different based on available data to be sampled)
    n_train = size(y_train)[end]
    n_valid = size(y_valid)[end]

    normalize_func(x, μ, σ) = (x .- μ) ./ (σ .+ ϵ_std)

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
        if use_running_stats
            # Keep track of all y-data to get better mean/std across entire sets of training data (running mean/std)
            map(y->push!(rstats, y), y_train[1,:])
            mean_y = mean(rstats)
            std_y = std(rstats)
        else
            mean_y, std_y = mean_and_std(y_train[1,:])
        end
        y_train[1,:] = normalize_func(y_train[1,:], mean_y, std_y)
        y_valid[1,:] = normalize_func(y_valid[1,:], mean_y, std_y)
    end

    verbose && @info "Data set size: $(n_train):$(n_valid) (training:validation)"

    if n_train < nn_params.batchsize
        batchsize = n_train
        @warn("Number of observations less than batch-size, decreasing the batch-size to $batchsize", maxlog=1)
    else
        batchsize = nn_params.batchsize
    end

    train_data = Flux.Data.DataLoader((x_train, y_train), batchsize=batchsize, shuffle=true)
    valid_data = Flux.Data.DataLoader((x_valid, y_valid), batchsize=batchsize, shuffle=true)

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


    verbose && @info "Using value weight of: $value_loss_weight"

    loss(x, y; w=value_loss_weight, info=Dict()) = begin
        local ỹ = f(x)
        n = size(ỹ,1)-1
        vmask = vcat(1, zeros(Int,n))
        if device == gpu
            vmask = Flux.CuArray(vmask)
        end
        pmask = 1 .- vmask
        v = vmask .* ỹ # value prediction
        𝐩 = pmask .* ỹ # policy prediction
        z = vmask .* y # true value
        π = pmask .* y # true policy vector

        value_loss = w*nn_params.loss_func(v, z)
        if use_kl_loss
            policy_loss = (1-w)*Flux.Losses.kldivergence(𝐩, π)
        else
            policy_loss = (1-w)*Flux.Losses.crossentropy(𝐩, π)
        end
        regularization = penalty()

        ignore_derivatives() do
            info[:value_loss] = value_loss
            info[:policy_loss] = policy_loss
            info[:regularization] = regularization
        end

        value_loss + policy_loss + regularization
    end

    opt = nn_params.optimizer(lr)
    θ = Flux.params(f)

    training_epochs = nn_params.training_epochs
    losses_train = []
    losses_train_value = []
    losses_train_policy = []
    losses_valid = []
    losses_valid_value = []
    losses_valid_policy = []
    accs_train = []
    accs_valid = []
    learning_curve = nothing
    checkpoint_loss_valid = Inf
    checkpoint_loss_valid_value = Inf
    checkpoint_loss_valid_policy = Inf
    checkpoint_loss_train = Inf
    checkpoint_loss_train_value = Inf
    checkpoint_loss_train_policy = Inf
    checkpoint_acc_valid = 0
    checkpoint_acc_train = 0
    f_checkpoint = f

    local loss_train = Inf
    local loss_valid = Inf
    local acc_train = 0
    local acc_valid = 0
    local checkpoint_epoch = Inf

    logging_fn(epoch, loss_train, loss_train_value, loss_train_policy, loss_valid, loss_valid_value, loss_valid_policy, acc_train, acc_valid; extra="", digits=5) = string("Epoch: ", epoch, "\t Loss Train: ", round(loss_train; digits), " [", round(loss_train_value; digits), ", ", round(loss_train_policy; digits), "]\t Loss Val: ", round(loss_valid; digits), " [", round(loss_valid_value; digits), ", ", round(loss_valid_policy; digits), "]\t|\t Sign Acc. Train: ", rpad(round(acc_train; digits), digits+2, '0'), "\t Sign Acc. Val: ", rpad(round(acc_valid; digits), digits+2, '0'), extra)

    function plot_training(e, training_epochs, losses_train, losses_train_value, losses_train_policy, losses_valid, losses_valid_value, losses_valid_policy, key)
        learning_curve = plot(xlims=(1, training_epochs), title="learning curve: $key")
        plot!(1:e, losses_train, label="training", c=1)
        plot!(1:e, losses_train_value, label="training (value)", c=1, ls=:dash)
        plot!(1:e, losses_train_policy, label="training (policy)", c=1, ls=:dot)
        plot!(1:e, losses_valid, label="validation", c=2)
        plot!(1:e, losses_valid_value, label="validation (value)", c=2, ls=:dash)
        plot!(1:e, losses_valid_policy, label="validation (policy)", c=2, ls=:dot)
        ylims!(0, ylims()[2])
        return learning_curve
    end

    # Batch calculate the loss, placing data on device
    function calc_loss(data; w=value_loss_weight, info=Dict())
        ℓ = 0
        total = 0
        for (x, y) in data
            local_info = Dict()
            ℓ += loss(device(x), device(y); w, info=local_info)
            merge!(+, info, local_info)
            total += size(y, 2)
        end
        for k in keys(info)
            info[k] = info[k]/total
        end
        return ℓ/total
    end

    function calc_sign_accuracy(data)
        matched = 0
        total = 0
        for (x, y) in data
            x = device(x)
            y = device(y)
            v = f(x)[1,:]
            g = y[1,:]
            matched += sum(sign.(v) .== sign.(g))
            total += length(v)
        end
        return matched/total
    end

    has_stopped_short = false
    final_stopped_epoch = -Inf

    verbose && @info "Beginning training $(size(x_train))"
    @conditional_time verbose for e in 1:training_epochs
        w = value_loss_weight
        for (x, y) in train_data
            # Only put batches on device
            x = device(x)
            y = device(y)
            _, back = Flux.pullback(() -> loss(x, y; w), θ)
            Flux.update!(opt, θ, back(1.0f0))
        end
        loss_train_info = Dict()
        loss_train = calc_loss(train_data; w, info=loss_train_info)
        loss_train_value = loss_train_info[:value_loss]
        loss_train_policy = loss_train_info[:policy_loss]
        loss_valid_info = Dict()
        loss_valid = calc_loss(valid_data; w, info=loss_valid_info)
        loss_valid_value = loss_valid_info[:value_loss]
        loss_valid_policy = loss_valid_info[:policy_loss]
        acc_train = calc_sign_accuracy(train_data)
        acc_valid = calc_sign_accuracy(valid_data)
        push!(losses_train, loss_train)
        push!(losses_train_value, loss_train_value)
        push!(losses_train_policy, loss_train_policy)
        push!(losses_valid, loss_valid)
        push!(losses_valid_value, loss_valid_value)
        push!(losses_valid_policy, loss_valid_policy)
        push!(accs_train, acc_train)
        push!(accs_valid, acc_valid)
        if verbose && e % nn_params.verbose_update_frequency == 0
            println(logging_fn(e, loss_train, loss_train_value, loss_train_policy, loss_valid, loss_valid_value, loss_valid_policy, acc_train, acc_valid))
        end
        if e % nn_params.checkpoint_frequency == 0
            checkpoint_condition = nn_params.checkpoint_validation_loss ? loss_valid < checkpoint_loss_valid : loss_train < checkpoint_loss_train
            if checkpoint_condition
                checkpoint_loss_valid = loss_valid
                checkpoint_loss_valid_value = loss_valid_value
                checkpoint_loss_valid_policy = loss_valid_policy
                checkpoint_loss_train = loss_train
                checkpoint_loss_train_value = loss_train_value
                checkpoint_loss_train_policy = loss_train_policy
                checkpoint_acc_valid = acc_valid
                checkpoint_acc_train = acc_train
                checkpoint_epoch = e
                f_checkpoint = deepcopy(f)
                verbose && println(logging_fn(e, loss_train, loss_train_value, loss_train_policy, loss_valid, loss_valid_value, loss_valid_policy, acc_train, acc_valid; extra=" [Checkpoint]"))
            end
        end

        if e % nn_params.verbose_plot_frequency == 0
            learning_curve = plot_training(e, training_epochs, losses_train, losses_train_value, losses_train_policy, losses_valid, losses_valid_value, losses_valid_policy, key)
            nn_params.save_plots && Plots.savefig(nn_params.plot_curve_filename)
            nn_params.display_plots && display(learning_curve)
        end

        if nn_params.stop_short && e - checkpoint_epoch > nn_params.stop_short_threshold
            @info "Stopping short at epoch $e"
            has_stopped_short = true
            final_stopped_epoch = e
            break
        end
    end

    if has_stopped_short && nn_params.verbose_plot_frequency != Inf
        learning_curve = plot_training(final_stopped_epoch, training_epochs, losses_train, losses_train_value, losses_train_policy, losses_valid, losses_valid_value, losses_valid_policy, key)
        nn_params.save_plots && Plots.savefig(nn_params.plot_curve_filename)
        nn_params.display_plots && display(learning_curve)
    end

    if nn_params.use_checkpoint
        # check final loss
        checkpoint_condition = nn_params.checkpoint_validation_loss ? loss_valid < checkpoint_loss_valid : loss_train < checkpoint_loss_train
        if checkpoint_condition
            checkpoint_loss_valid = loss_valid
            checkpoint_loss_valid_value = loss_valid_value
            checkpoint_loss_valid_policy = loss_valid_policy
            checkpoint_loss_train = loss_train
            checkpoint_loss_train_value = loss_train_value
            checkpoint_loss_train_policy = loss_train_policy
            checkpoint_acc_valid = acc_valid
            checkpoint_acc_train = acc_train
            checkpoint_epoch = training_epochs
            f_checkpoint = deepcopy(f)
        end
        verbose && println(logging_fn(checkpoint_epoch, checkpoint_loss_train, checkpoint_loss_train_value, checkpoint_loss_train_policy, checkpoint_loss_valid, checkpoint_loss_valid_value, checkpoint_loss_valid_policy, checkpoint_acc_train, checkpoint_acc_valid; extra=" [Final network checkpoint]"))
        f = f_checkpoint
    end

    # Place network on the CPU (better GPU memory conservation when doing parallelized inference)
    f = cpu(f)

    if normalize_input
        # Add normalization input layer
        unnormalize_x = x -> (x .* std_x) .+ mean_x
        normalize_x = x -> (x .- mean_x) ./ (std_x .+ ϵ_std)
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

    value_model = value_data = nothing

    if nn_params.verbose_plot_frequency != Inf
        value_distribution = nothing
        try
            @warn("Data size may be too much to run it entirely through the network...")
            value_model = normalize_input ? f(unnormalize_x(cpu(x_valid)))[1,:] : f(cpu(x_valid))[1,:]
            value_data = normalize_output ? unnormalize_y(cpu(y_valid))[1,:] : cpu(y_valid)[1,:]

            value_distribution = Plots.histogram(value_model, alpha=0.5, label="model", c=:gray, title="values: $key")
            Plots.histogram!(value_data, alpha=0.5, label="data", c=:navy)
            nn_params.save_plots && Plots.savefig(nn_params.plot_value_distribution_filename)
            nn_params.display_plots && display(value_distribution)

            plot_bias(value_model, value_data)
            nn_params.save_plots && Plots.savefig(nn_params.plot_validation_bias_filename)
            nn_params.display_plots && display(Plots.title!("validation data"))

            value_model_training = normalize_input ? f(unnormalize_x(cpu(x_train)))[1,:] : f(cpu(x_train))[1,:]
            value_data_training = normalize_output ? unnormalize_y(cpu(y_train))[1,:] : cpu(y_train)[1,:]
            plot_bias(value_model_training, value_data_training)
            nn_params.save_plots && Plots.savefig(nn_params.plot_training_bias_filename)
            nn_params.display_plots && display(Plots.title!("training data"))
        catch err
            @warn "Error in plotting learning curve and value distribution: $err"
        end
    end

    # Save training results
    if !isnothing(results) && isa(results, Dict)
        results[key] = Dict(
            "losses_train" => losses_train,
            "losses_train_value" => losses_train_value,
            "losses_train_policy" => losses_train_policy,
            "losses_valid" => losses_valid,
            "losses_valid_value" => losses_valid_value,
            "losses_valid_policy" => losses_valid_policy,
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
        if device == gpu
            Flux.CUDA.reclaim()
        end
    end

    return f
end


"""
Get belief representation for network input, add batch dimension for Flux.
"""
function network_input(belief)
    b = Float32.(input_representation(belief))
    return Flux.unsqueeze(b; dims=ndims(b)+1) # add extra single dimension (batch)
end


"""
Evaluate the neural network `f` using the `belief` as input, return both the predicted value and policy vector.
Note, inference is done on the CPU given a single input.
"""
network_lookup(policy::BetaZeroPolicy, belief) = network_lookup(policy.surrogate, belief)
function network_lookup(f::Union{Chain,EnsembleNetwork}, belief)
    x = network_input(belief)
    y = cpu(f(x)) # evaluate network `f`
    return y[1], y[2:end] # [v, p...] NOTE: This ordering is different than the paper which defines (𝐩, v)
end


"""
Evaluate the neural network `f` using the `belief` as input, return the predicted value.
Note, inference is done on the CPU given a single input.
"""
value_lookup(policy::BetaZeroPolicy, belief) = value_lookup(policy.surrogate, belief)
value_lookup(f::Union{Chain,EnsembleNetwork}, belief) = network_lookup(f, belief)[1] # (v, p)
POMDPs.value(policy::BetaZeroPolicy, belief) = value_lookup(policy, belief)
POMDPs.value(f::Union{Chain,EnsembleNetwork}, belief) = value_lookup(f, belief)

"""
Evaluate the ensemble neural network `f` using the `belief` as input, return variance of predicted values.
Note, inference is done on the CPU given a single input.
"""
uncertainty_lookup(policy::BetaZeroPolicy, belief) = uncertainty_lookup(policy.surrogate, belief)
function uncertainty_lookup(f::EnsembleNetwork, belief)
    x = network_input(belief)
    μ, σ = cpu(f(x,return_std=true)) # evaluate network `f`. EnsembleNetwork is capable of returning μ, σ if return_std is set to true
    return σ # suhastag here's where we could change standard deviation or variance
end
"""
Evaluate the Neural network `f` using the `belief` as input, return variance of predicted values.
This should work, as `f` is in train mode and employs dropout. Then there should be nonzero variance of 
predicted values
Note, inference is done on the CPU given a single input.
"""
function uncertainty_lookup(f::Chain, belief)
    x = network_input(belief)
    preds = []
    for i in 1:20 # TODO: change from 20!!
        preds.push!(cpu(f(x))) # evaluate network `f`
    end
    return std(preds) # suhastag here's where we could change standard deviation or variance
end
"""
Evaluate the neural network `f` using the `belief` as input, return the predicted policy vector.
Note, inference is done on the CPU given a single input.
"""
policy_lookup(policy::BetaZeroPolicy, belief) = policy_lookup(policy.surrogate, belief)
policy_lookup(f::Union{Chain,EnsembleNetwork}, belief) = network_lookup(f, belief)[2] # (v, p)


"""
Use predicted policy vector to sample next action.
"""
function next_action(problem::Union{BeliefMDP, POMDP}, belief, f::Union{Chain,EnsembleNetwork}, nn_params::BetaZeroNetworkParameters, bnode)
    Ab = POMDPs.actions(problem, belief)

    if nn_params.use_prioritized_action_selection
        p = policy_lookup(f, belief)
        A = POMDPs.actions(problem)

        # Match indices of (potentially) reduced belief-dependent action space to get correctly associated probabilities from the network
        if length(A) != length(Ab)
            idx = Vector{Int}(undef, length(Ab))
            for (i,a) in enumerate(A)
                for (j,ab) in enumerate(Ab)
                    if a == ab
                        idx[j] = i
                        break
                    end
                end
            end
            p = p[idx]
        end

        # Zero-out already tried actions
        if nn_params.zero_out_tried_actions
            action_indices = bnode.tree.children[bnode.index]
            tried_actions = bnode.tree.a_labels[action_indices]
            if !isempty(tried_actions) && length(tried_actions) != length(Ab)
                for (i,a) in enumerate(tried_actions)
                    for (j,ab) in enumerate(Ab)
                        if a == ab
                            p[j] = 1e-6 # zero-out
                            break
                        end
                    end
                end
            end
        end

        p = normalize(p, 1) # re-normalize to sum to 1

        if nn_params.use_epsilon_greedy && rand() < nn_params.ϵ_greedy
            return rand(Ab)
        else
            if nn_params.next_action_return_argmax
                return Ab[argmax(p)]
            else
                return rand(SparseCat(Ab, p))
            end
        end
    else
        # Sample randomly from the action space
        return rand(Ab)
    end
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
