@with_kw mutable struct GaussianProcess
    gp::GPE # Gaussian process object
    predict::Function = gp_mean(gp) # Gaussian process prediction function
end


@with_kw mutable struct GPSurrogate
    fV::GaussianProcess # Gaussian process object for value function
    fP::Vector{GaussianProcess} # Gaussian process object for policy
end


# Calling objects as functions
(gp::GaussianProcess)(x) = gp.predict(x)
(gps::Vector{GaussianProcess})(x) = softmax(map(gp->gp(x), gps))

function (surrogate::GPSurrogate)(x)
    ṽ = surrogate.fV(x)
    p̃ = surrogate.fP(x)
    return vcat(ṽ, p̃)
end


"""
Return function to compute lower confidence bound (LCB) from the Gaussian process surrogate.
"""
function gp_lcb(gp; λ=0.1)
    return xy->begin
        ỹ = predict_f(gp, Float64.(reshape(xy, (:,1))))
        μ, σ² = ỹ[1][1], ỹ[2][1]
        return μ - λ*sqrt(σ²)
    end
end


"""
Return function to compute mean from the Gaussian process surrogate.
"""
gp_mean(gp) = xy->predict_f(gp, Float64.(reshape(xy, (:,1))))[1][1]


"""
Return function to compute std (uncertainty) from the Gaussian process surrogate.
"""
gp_std(gp) = xy->sqrt(predict_f(gp, Float64.(reshape(xy, (:,1))))[2][1])


"""
Initialize surrogate model (Gaussian process).
"""
function initialize_gaussian_process(solver::BetaZeroSolver)
    gp_params = solver.gp_params
    input_size = gp_params.input_size
    kernel_v = gp_params.kernel_v
    kernel_p = gp_params.kernel_p
    λ = gp_params.λ_lcb

    gpV = GP(zeros(input_size..., 1), zeros(1), MeanZero(), kernel_v)
    predictV = gp_params.use_lcb_initial ? gp_lcb(gpV; λ) : gp_mean(gpV)
    fV = GaussianProcess(gp=gpV, predict=predictV)

    fP = GaussianProcess[]
    as = actions(solver.pomdp)
    for _ in as
        gpP = GP(zeros(input_size..., 1), zeros(1), MeanZero(), kernel_p)
        predictP = gp_params.use_lcb_initial ? gp_lcb(gpP; λ) : gp_mean(gpP)
        fPi = GaussianProcess(gp=gpP, predict=predictP)
        push!(fP, fPi)
    end

    return GPSurrogate(; fV, fP)
end


"""
Train policy & value Gaussian procces `f` using the latest `data` generated from online tree search (MCTS).
This optimizes the Gaussian processes (value and policy surrogates) by minimizing error on the validation data.
"""
function train(f::GPSurrogate, solver::BetaZeroSolver; verbose::Bool=false, results=nothing)
    gp_params = solver.gp_params
    opt_iterations = gp_params.opt_iterations
    tune_individual_action_gps = gp_params.tune_individual_action_gps

    n_train = Int(gp_params.n_samples ÷ (1/gp_params.training_split))
    n_valid = gp_params.n_samples - n_train

    data_train = sample_data(solver.data_buffer_train, n_train) # sample from last `n_buffer` simulations.
    data_valid = sample_data(solver.data_buffer_valid, n_valid) # sample from last `n_buffer` simulations.

    x_train = data_train.X'
    x_valid = data_valid.X'
    y_train_V = data_train.Y[1, :]
    y_valid_V = data_valid.Y[1, :]
    y_train_P = data_train.Y[2:end, :]
    y_valid_P = data_valid.Y[2:end, :]

    x_train_gp = Float64.(x_train)'

    if gp_params.optimize
        gp_value_error(x, z, fV) = Flux.Losses.mae(fV(x), z)

        function value_opt(θ)
            ℓ, σ = θ
            (ℓ ≤ 0 || σ ≤ 0) && return Inf
            kernel_v = f.fV.gp.kernel
            if hasproperty(kernel_v, :ℓ2)
                kernel_v.ℓ2 = ℓ^2
            elseif hasproperty(kernel_v, :ℓ)
                kernel_v.ℓ = ℓ
            else
                error("Support for this GP kernel is not yet implemented.")
            end
            kernel_v.σ2 = σ^2
            fV = fit_value_gp(solver, f, x_train_gp, y_train_V; kernel=kernel_v)
            return mean(gp_value_error(xi, zi, fV) for (xi,zi) in zip(eachrow(x_valid), y_valid_V))
        end

        res_value = optimize(value_opt, [1.0, 1.0], NelderMead(), Optim.Options(show_trace=false, iterations=opt_iterations÷2))

        # Apply tuned parameters to GP
        ℓ, σ = Optim.minimizer(res_value)
        kernel_v = f.fV.gp.kernel
        if hasproperty(kernel_v, :ℓ2)
            kernel_v.ℓ2 = ℓ^2
        elseif hasproperty(kernel_v, :ℓ)
            kernel_v.ℓ = ℓ
        else
            error("Support for this GP kernel is not yet implemented.")
        end
        kernel_v.σ2 = σ^2

        fV = fit_value_gp(solver, f, x_train_gp, y_train_V; kernel=kernel_v)

        if solver.verbose
            mean_value_error_train = mean(gp_value_error(xi, zi, fV) for (xi,zi) in zip(eachrow(x_train), y_train_V))
            mean_value_error_valid = mean(gp_value_error(xi, zi, fV) for (xi,zi) in zip(eachrow(x_valid), y_valid_V))
            @info "GP mean value error (train): $mean_value_error_train"
            @info "GP mean value error (validation): $mean_value_error_valid"
        end
    else
        fV = fit_value_gp(solver, f, x_train_gp, y_train_V)
    end

    if gp_params.optimize
        gp_policy_error(x, π, fP) = Flux.Losses.crossentropy(fP(x), π)

        function policy_opt(θ)
            if tune_individual_action_gps
                # Individual kernel parameters
                kernels_P = Vector(undef, length(f.fP))
                i = 1
                for ai in eachindex(f.fP)
                    ℓ = θ[i]
                    σ = θ[i+1]
                    (ℓ ≤ 0 || σ ≤ 0) && return Inf
                    i += 2
                    kernel_pi = f.fP[ai].gp.kernel
                    if hasproperty(kernel_pi, :ℓ2)
                        kernel_pi.ℓ2 = ℓ^2
                    elseif hasproperty(kernel_pi, :ℓ)
                        kernel_pi.ℓ = ℓ
                    else
                        error("Support for this GP kernel is not yet implemented.")
                    end
                    kernel_pi.σ2 = σ^2
                    kernels_P[ai] = kernel_pi
                end
            else
                # Shared kernel parameters
                ℓ, σ = θ
                (ℓ ≤ 0 || σ ≤ 0) && return Inf
                kernels_P = Vector(undef, length(f.fP))
                for ai in eachindex(f.fP)
                    kernel_pi = f.fP[ai].gp.kernel
                    if hasproperty(kernel_pi, :ℓ2)
                        kernel_pi.ℓ2 = ℓ^2
                    elseif hasproperty(kernel_pi, :ℓ)
                        kernel_pi.ℓ = ℓ
                    else
                        error("Support for this GP kernel is not yet implemented.")
                    end
                    kernel_pi.σ2 = σ^2
                    kernels_P[ai] = kernel_pi
                end
            end

            fP = fit_policy_gp(solver, f, x_train_gp, y_train_P; kernels=kernels_P)
            return mean(gp_policy_error(xi, πi, fP) for (xi,πi) in zip(eachrow(x_valid), y_valid_P))
        end

        initial_policy_θ = tune_individual_action_gps ? repeat([1.0, 1.0], length(f.fP)) : [1.0, 1.0]
        res_policy = optimize(policy_opt, initial_policy_θ, NelderMead(), Optim.Options(show_trace=false, iterations=opt_iterations÷2))

        # Apply tuned parameters to GP
        if tune_individual_action_gps
            kernels_P = Vector(undef, length(f.fP))
            i = 1
            for ai in eachindex(f.fP)
                ℓ = θP[i]
                σ = θP[i+1]
                i += 2
                kernel_pi = f.fP[ai].gp.kernel
                if hasproperty(kernel_pi, :ℓ2)
                    kernel_pi.ℓ2 = ℓ^2
                elseif hasproperty(kernel_pi, :ℓ)
                    kernel_pi.ℓ = ℓ
                else
                    error("Support for this GP kernel is not yet implemented.")
                end
                kernel_pi.σ2 = σ^2
                kernels_P[ai] = kernel_pi
            end
        else
            ℓ, σ = Optim.minimizer(res_policy)
            kernels_P = Vector(undef, length(f.fP))
            for ai in eachindex(f.fP)
                kernel_pi = f.fP[ai].gp.kernel
                if hasproperty(kernel_pi, :ℓ2)
                    kernel_pi.ℓ2 = ℓ^2
                elseif hasproperty(kernel_pi, :ℓ)
                    kernel_pi.ℓ = ℓ
                else
                    error("Support for this GP kernel is not yet implemented.")
                end
                kernel_pi.σ2 = σ^2
                kernels_P[ai] = kernel_pi
            end
        end

        fP = fit_policy_gp(solver, f, x_train_gp, y_train_P; kernels=kernels_P)

        if solver.verbose
            mean_policy_error_train = mean(gp_policy_error(xi, πi, fP) for (xi,πi) in zip(eachrow(x_train), y_train_P))
            mean_policy_error_valid = mean(gp_policy_error(xi, πi, fP) for (xi,πi) in zip(eachrow(x_valid), y_valid_P))
            @info "GP mean policy error (train): $mean_policy_error_train"
            @info "GP mean policy error (validation): $mean_policy_error_valid"
        end
    else
        fP = fit_policy_gp(solver, f, x_train_gp, y_train_P)
    end

    if gp_params.verbose_plot
        plot_gp(fV, fP, solver, x_train, x_valid, y_train_V, y_valid_V, y_train_P, y_valid_P)
    end

    return GPSurrogate(; fV, fP)
end


"""
Fit a Gaussian procces to the training data for the value function surrogate.
"""
function fit_value_gp(solver::BetaZeroSolver, f::GPSurrogate, x_train, y_train; kernel=f.fV.gp.kernel)
    gp_params = solver.gp_params

    mean_fn = MeanZero()
    λ = gp_params.λ_lcb

    gp = GP(x_train, y_train, mean_fn, kernel)
    predict = gp_params.use_lcb ? gp_lcb(gp; λ) : gp_mean(gp)
    return GaussianProcess(; gp, predict)
end


"""
Fit a Gaussian procces to the training data for the policy surrogate.
"""
function fit_policy_gp(solver::BetaZeroSolver, f::GPSurrogate, x_train, y_train; kernels=[fPi.gp.kernel for fPi in f.fP])
    gp_params = solver.gp_params

    as = actions(solver.pomdp)
    num_actions = length(as)
    fP = GaussianProcess[]
    mean_fn_P = MeanConst(1/num_actions)

    for i in 1:num_actions
        kernel = kernels[i]
        λ = gp_params.λ_lcb
        y_train_Pi = y_train[i,:]

        gp_Pi = GP(x_train, y_train_Pi, mean_fn_P, kernel)
        predict = gp_params.use_lcb ? gp_lcb(gp_Pi; λ) : gp_mean(gp_Pi)
        fPi = GaussianProcess(; gp=gp_Pi, predict)
        push!(fP, fPi)
    end

    return fP::Vector{GaussianProcess}
end


"""
Gaussian process look-up/evaluation.
"""
function value_lookup(f::GPSurrogate, belief)
    x = Float64.(input_representation(belief))
    y = f(x) # evaluate GP `f`
    return y[1]
end


"""
Evaluate the Gaussian process `f` using the `belief` as input, return the predicted policy vector.
"""
function policy_lookup(f::GPSurrogate, belief)
    x = Float64.(input_representation(belief))
    y = f(x) # evaluate GP `f`
    return y[2:end]
end


"""
Use predicted policy vector to sample next action.
"""
function next_action(problem::Union{BeliefMDP, POMDP}, belief, f::GPSurrogate)
    p = policy_lookup(f, belief)
    as = POMDPs.actions(problem) # TODO: actions(problem, belief) ??
    return rand(SparseCat(as, p))
end


"""
Plot value and policy Gaussian process surrogates.
"""
function plot_gp(fV::GaussianProcess, fP::Vector{GaussianProcess}, solver::BetaZeroSolver, x_train, x_valid, y_train_V, y_valid_V, y_train_P, y_valid_P)
    gp_params = solver.gp_params

    gp_y_valid = [fV(x) for x in eachrow(x_valid)]
    BetaZero.plot_bias(gp_y_valid, y_valid_V)
    Plots.title!("GP validation")
    display(plot!())

    gp_y_train = [fV(x) for x in eachrow(x_train)]
    BetaZero.plot_bias(gp_y_train, y_train_V)
    Plots.title!("GP training")
    display(plot!())

    gp_plts = []
    x1min, x1max = minimum(x_train[:,1]), maximum(x_train[:,1])
    x2min, x2max = minimum(x_train[:,2]), maximum(x_train[:,2])
    x1min2, x1max2 = minimum(x_valid[:,1]), maximum(x_valid[:,1])
    x2min2, x2max2 = minimum(x_valid[:,2]), maximum(x_valid[:,2])
    x1min = min(x1min, x1min2)
    x1max = max(x1max, x1max2)
    x2min = min(x2min, x2min2)
    x2max = max(x2max, x2max2)

    pltX = range(x2min, x2max, length=100)
    pltY = range(x1min, x1max, length=100)

    ymin, ymax = minimum(y_train_V), maximum(y_train_V)
    ymin2, ymax2 = minimum(y_valid_V), maximum(y_valid_V)

    gp_ỹ = [fV([x,y]) for x in pltY, y in pltX] # NOTE x-y flip

    cmap_data = shifted_colormap([min(ymin,ymin2), max(ymax,ymax2)])
    cmap_model = shifted_colormap(gp_ỹ)

    if gp_params.verbose_show_value_plot
        for training in [true, false]
            plot(size=(800,400), legend=:outerbottomleft)

            # NOTE: Switch (μ on y_train_V-axis, σ on x-axis)
            heatmap!(pltX, pltY, gp_ỹ, c=cmap_model)

            if training
                scatter!(x_train[:,2], x_train[:,1], cmap=[get(cmap_data, normalize01(yi,y_train_V)) for yi in y_train_V], marker=:square, msc=:gray, alpha=0.5, label="–training-")
            else
                scatter!(x_valid[:,2], x_valid[:,1], cmap=[get(cmap_data, normalize01(yi,y_valid_V)) for yi in y_valid_V], marker=:circle, msc=:black, alpha=0.5, label="validation")
            end

            Plots.xlabel!("\$\\sigma(b)\$")
            Plots.ylabel!("\$\\mu(b)\$")
            Plots.title!(training ? "training" : "validation")
            push!(gp_plts, plot!())
        end
        map(display, gp_plts)
    end

    # Individual action heatmap
    if gp_params.verbose_show_individual_action_plots
        as = actions(solver.pomdp)
        for (i,fPi) in enumerate(fP)
            gp_P̃i = [fPi([x,y]) for x in pltY, y in pltX] # NOTE x-y flip
            Plots.heatmap(pltX, pltY, gp_P̃i, title="action: $(as[i])", c=:viridis) |> display
        end
    end

    if gp_params.verbose_show_policy_plot
        as = actions(solver.pomdp)
        num_actions = length(as)
        gp_P̃ = [as[argmax(softmax(map(fPi->fPi([x,y]), fP)))] for x in pltY, y in pltX] # NOTE x-y flip
        Plots.heatmap(pltX, pltY, gp_P̃, title="policy", c=palette(:viridis,num_actions)) |> display
    end
end