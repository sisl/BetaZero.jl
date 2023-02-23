@with_kw mutable struct GPSurrogate
    gp::GPE # Gaussian proccess object
    predict::Function = gp_mean(gp) # Gaussian proccess prediction function
    optimize::Bool = false # Indicate if GP params should be optimized to the training data (generally keep at `false`)
end


# Calling objects as functions
(surrogate::GPSurrogate)(x) = surrogate.predict(x)


"""
Return function to compute lower confidence bound (LCB) from the Gaussian proccess surrogate.
"""
function gp_lcb(gp; λ=0.1)
    return xy->begin
        ỹ = predict_f(gp, Float64.(reshape(xy, (:,1))))
        μ, σ² = ỹ[1][1], ỹ[2][1]
        return μ - λ*sqrt(σ²)
    end
end


"""
Return function to compute mean from the Gaussian proccess surrogate.
"""
gp_mean(gp) = xy->predict_f(gp, Float64.(reshape(xy, (:,1))))[1][1]


"""
Return function to compute std (uncertainty) from the Gaussian proccess surrogate.
"""
gp_std(gp) = xy->sqrt(predict_f(gp, Float64.(reshape(xy, (:,1))))[2][1])


"""
Initialize surrogate model (Gaussian process).
"""
function initialize_gaussian_proccess(solver::BetaZeroSolver)
    gp_params = solver.gp_params
    input_size = gp_params.input_size
    kernel = gp_params.kernel
    λ = gp_params.λ_lcb
    optimize = gp_params.optimize

    gp = GP(zeros(input_size..., 1), zeros(1), MeanZero(), kernel)
    predict = gp_params.use_lcb_initial ? gp_lcb(gp; λ) : gp_mean(gp)

    return GPSurrogate(; gp, predict, optimize)
end


"""
Train policy & value Gaussian procces `f` using the latest `data` generated from online tree search (MCTS).
"""
function train(f::GPSurrogate, solver::BetaZeroSolver; verbose::Bool=false, results=nothing)
    gp_params = solver.gp_params

    n_train = Int(gp_params.n_samples ÷ (1/gp_params.training_split))
    n_valid = gp_params.n_samples - n_train

    data_train = sample_data(solver.data_buffer_train, n_train) # sample from last `n_buffer` simulations.
    data_valid = sample_data(solver.data_buffer_valid, n_valid) # sample from last `n_buffer` simulations.

    x_train = data_train.X'
    x_valid = data_valid.X'
    y_train = data_train.Y[:]
    y_valid = data_valid.Y[:]

    x_train_gp = Float64.(x_train)'
    gp_mean = MeanConst(Float64(mean(y_train)))
    kernel = f.gp.kernel
    optimize = f.optimize
    λ = gp_params.λ_lcb

    gp = GP(x_train_gp, y_train, gp_mean, kernel)
    optimize && @suppress optimize!(gp, GaussianProcesses.NelderMead())
    predict = gp_params.use_lcb ? gp_lcb(gp; λ) : gp_mean(gp)
    f = GPSurrogate(; gp, predict, optimize)

    if gp_params.verbose_plot
        gp_y_train = [f(x) for x in eachrow(x_train)]
        BetaZero.plot_bias(gp_y_train, y_train)
        Plots.title!("GP training")
        display(plot!())
    
        gp_y_valid = [f(x) for x in eachrow(x_valid)]
        BetaZero.plot_bias(gp_y_valid, y_valid)
        Plots.title!("GP validation")
        display(plot!())
    end

    if solver.verbose
        gp_error(x, y) = (y - f(x))^2
        mean_error_train = mean(gp_error(xi, yi) for (xi,yi) in zip(eachrow(x_train), y_train))
        mean_error_valid = mean(gp_error(xi, yi) for (xi,yi) in zip(eachrow(x_valid), y_valid))
        @info "GP mean error (train): $mean_error_train"
        @info "GP mean error (validation): $mean_error_valid"
    end

    return f::GPSurrogate
end


"""
Gaussian process look-up/evaluation.
"""
function value_lookup(belief, f::GPSurrogate)
    x = Float64.(input_representation(belief))
    y = f(x) # evaluate GP `f`
    return length(y) == 1 ? y[1] : y
end
