"""
Value function heatmaps (when applicable for scalar μ and σ)
"""
function value_plot(policy::BetaZeroPolicy;
                    xrange=range(0, 5, length=100),
                    yrange=range(-20, 20, length=100),
                    flip_axes::Bool=true,
                    xlabel="\$\\sigma(b)\$",
                    ylabel="\$\\mu(b)\$",
                    title="value")

    Yv = (x,y)->policy.surrogate(Float32.([x y])')[1]

    if flip_axes
        # mean on the y-axis, std on the x-axis
        Ydata = [Yv(y,x) for y in yrange, x in xrange] # Note x-y input flip
    else
        # mean on the x-axis, std on the y-axis
        xlabel, ylabel = ylabel, xlabel
        xrange, yrange = yrange, xrange
        Ydata = [Yv(x,y) for y in yrange, x in xrange]
    end
    cmap = shifted_colormap(Ydata)

    return Plots.heatmap(xrange, yrange, Ydata, xlabel=xlabel, ylabel=ylabel, title=title, cmap=cmap)
end


"""
Policy heatmaps (when applicable for scalar μ and σ and discrete actions)
"""
function policy_plot(pomdp::POMDP, policy::BetaZeroPolicy;
                    xrange=range(0, 5, length=100),
                    yrange=range(-20, 20, length=100),
                    flip_axes::Bool=true,
                    color=:viridis,
                    xlabel="\$\\sigma(b)\$",
                    ylabel="\$\\mu(b)\$",
                    title="policy")

    as = actions(pomdp)
    Yπ = (x,y)->as[argmax(policy.surrogate(Float32.([x y])')[2:end])]

    if flip_axes
        # mean on the y-axis, std on the x-axis
        Ydata = [Yπ(y,x) for y in yrange, x in xrange] # Note x-y input flip
    else
        # mean on the x-axis, std on the y-axis
        xlabel, ylabel = ylabel, xlabel
        xrange, yrange = yrange, xrange
        Ydata = [Yπ(x,y) for y in yrange, x in xrange]
    end

    return Plots.heatmap(xrange, yrange, Ydata, xlabel=xlabel, ylabel=ylabel, title=title, cmap=palette(color, length(as)))
end


"""
Plot combined value and policy plots.
"""
value_and_policy_plot(pomdp::POMDP, policy::BetaZeroPolicy; kwargs...) = plot(value_plot(policy; kwargs...), policy_plot(pomdp, policy; kwargs...), layout=2, size=(1000,300), margin=5Plots.mm)


"""
Plot combined value, policy, and uncertainty plots.
"""
function value_policy_uncertainty_plot(pomdp::POMDP, policy::BetaZeroPolicy; kwargs...)
    plt_value = value_plot(policy; kwargs...)
    plt_policy = policy_plot(pomdp, policy; kwargs...)
    plt_value_uncertainty = uncertainty_plot(policy; is_value=true, kwargs...)
    plt_policy_uncertainty = uncertainty_plot(policy; is_value=false, kwargs...)
    return plot(plt_value, plt_policy, plt_value_uncertainty, plt_policy_uncertainty, layout=4, size=(1000,600), margin=5Plots.mm)
end


"""
Plot uncertainty from ensemble surrogate of the value or policy function (when applicable for scalar μ and σ)
"""
function uncertainty_plot(policy::BetaZeroPolicy;
                          is_value::Bool=true, # value or policy uncertainty plots
                          xrange=range(0, 5, length=100),
                          yrange=range(-20, 20, length=100),
                          flip_axes::Bool=true,
                          cmap=:viridis,
                          xlabel="\$\\sigma(b)\$",
                          ylabel="\$\\mu(b)\$",
                          title="$(is_value ? "value" : "policy") (uncertainty)")

    if !isa(policy.surrogate, EnsembleNetwork)
        error("Cannot plot uncertainty using the surrogate $(typeof(policy.surrogate))")
    end

    if is_value
        Y = (x,y)->policy.surrogate(Float32.([x y])', return_std=true)[2][1]
    else
        Y = (x,y)->begin
            μ, σ = policy.surrogate(Float32.([x y])', return_std=true)
            μp, pσ = μ[2:end], σ[2:end]
            # Use σ from selected action
            return pσ[argmax(μp)]
        end
        # Y = (x,y)->sum(policy.surrogate(Float32.([x y])', return_std=true)[2][2:end]) # use sum of σ across actions
    end

    if flip_axes
        # mean on the y-axis, std on the x-axis
        Ydata = [Y(y,x) for y in yrange, x in xrange] # Note x-y input flip
    else
        # mean on the x-axis, std on the y-axis
        xlabel, ylabel = ylabel, xlabel
        xrange, yrange = yrange, xrange
        Ydata = [Y(x,y) for y in yrange, x in xrange]
    end

    return Plots.heatmap(xrange, yrange, Ydata, xlabel=xlabel, ylabel=ylabel, title=title, cmap=cmap)
end


"""
Plot holdout and performance metrics over iterations (core for `plot_accuracy` and `plot_returns`).
"""
function plot_metric(solver::BetaZeroSolver;
                     metric::Symbol=:accuracies,
                     xaxis_simulations=false,
                     xlabel=xaxis_simulations ? "total simulations" : "iteration",
                     ylabel="accuracy",
                     title=ylabel,
                     include_holdout=false,
                     include_data_gen=true,
                     relative_to_optimal=true, # NOTE! TODO: parameterize in solver
                     apply_rolling_mean=false, # NOTE! TODO: parameterize in solver
                     apply_smoothing=true, # TODO: Parameterize
                     use_percentiles=true, # TODO!!!
                     expert_results=nothing, # [mean, std]
                     expert_label="expert")

    if metric == :accuracies
        if include_holdout
            ho_results = [mean_and_std(getfield(m, metric)) for m in solver.holdout_metrics]
            ho_μ = first.(ho_results)
            ho_σ = last.(ho_results)
            if apply_rolling_mean
                ho_μ = rolling_mean(ho_μ)
            end
        end
        pm_x_over_time = accuracy_over_time
    elseif metric == :returns
        if include_holdout
            if relative_to_optimal
                ho_μ = [mean([m.returns[i] - m.optimal_returns[i] for i in  eachindex(m.returns)]) for m in solver.holdout_metrics]
                ho_σ = [std([m.returns[i] - m.optimal_returns[i] for i in eachindex(m.returns)]) for m in solver.holdout_metrics]
            else
                ho_μ = [m.mean for m in solver.holdout_metrics]
                ho_σ = [m.std for m in solver.holdout_metrics]
            end
            if apply_rolling_mean
                ho_μ = rolling_mean(ho_μ)
            end
        end
        if relative_to_optimal
            pm_x_over_time = relative_returns_over_time
            ylabel = "(relative) $ylabel"
            title = "(relative) $title"
        else
            pm_x_over_time = returns_over_time
        end
    else
        error("`plot_metric` not yet defined for metric $metric")
    end

    if include_holdout
        n_ho = solver.params.n_holdout
        if apply_rolling_mean
            ho_stderr = rolling_stderr(ho_μ)
            ho_stderr[1] = 0 # NaN
        else
            ho_stderr = ho_σ ./ sqrt(n_ho)
        end
    end

    pm_results = pm_x_over_time(solver, mean_and_std)
    pm_μ = first.(pm_results)
    pm_σ = last.(pm_results)
	n_pm = solver.params.n_data_gen
    pm_stderr = pm_σ ./ sqrt(n_pm)

    if apply_rolling_mean
        pm_μ = rolling_mean(pm_μ)
        pm_stderr = rolling_stderr(pm_μ)
        pm_stderr[1] = 0 # NaN
    end

    plot(title=title, ylabel=ylabel, xlabel=xlabel, legend=:bottomright)

    local pm_X = [1]
    if include_data_gen
        if xaxis_simulations
            pm_X = count_simulations_accumulated(solver; zero_start=true)[1:end-1]
            pm_X = pm_X[1:length(pm_μ)]
        else
            pm_X = eachindex(pm_μ)
        end
        pm_color = :crimson
        pm_label = "BetaZero (data collection)"
        if apply_smoothing && !apply_rolling_mean && length(pm_μ) > 1
            plot!(pm_X, pm_μ, alpha=0.3, color=pm_color, label=false)
            plot!(pm_X, smooth(pm_μ), color=pm_color, label=pm_label, linewidth=2, fillalpha=0.2, ribbon=smooth(pm_stderr), alpha=0.8)
        else
            plot!(pm_X, pm_μ, ribbon=pm_stderr, fillalpha=0.2, lw=2, marker=length(pm_μ)==1, label=pm_label, c=pm_color, alpha=0.5)
        end
    end

    local ho_X = [1]
    if include_holdout
        if xaxis_simulations
            ho_X = count_simulations_accumulated(solver; zero_start=true)
            ho_X = ho_X[1:length(ho_μ)]
        else
            ho_X = eachindex(ho_μ)
        end
        ho_color = :darkgreen
        ho_label = "BetaZero (holdout)"
        if apply_smoothing && !apply_rolling_mean && length(ho_μ) > 1
            plot!(ho_X, ho_μ, alpha=0.3, color=ho_color, label=false)
            plot!(ho_X, smooth(ho_μ), color=ho_color, label=ho_label, linewidth=2, fillalpha=0.2, ribbon=smooth(ho_stderr), alpha=0.8)
        else
            plot!(ho_X, ho_μ, ribbon=ho_stderr, fillalpha=0.1, lw=2, marker=length(ho_μ)==1, label=ho_label, c=ho_color, alpha=0.5)
        end

    end

    if !isnothing(expert_results)
        expert_means, expert_stderr = expert_results
        hline!([expert_means...], ribbon=expert_stderr, fillalpha=0.2, ls=:dash, lw=1, c=:black, label=expert_label)
    end

    # Currently running, make x-axis span entire iteration domain
    if xaxis_simulations
        xlims!(min(ho_X[1], pm_X[1]), count_simulations(solver))
    else
        # TODO: Handle `resume`
        additional_iterations = 0 # 20
        xlims!(1, max(solver.params.n_iterations, length(pm_X)))
    end

    if metric == :accuracies
        ylims!(ylims()[1], 1.01)
    end

    return plot!()
end

plot_accuracy(solver::BetaZeroSolver; kwargs...) = plot_metric(solver; metric=:accuracies, ylabel="accuracy", kwargs...)
plot_returns(solver::BetaZeroSolver; kwargs...) = plot_metric(solver; metric=:returns, ylabel="returns", kwargs...)
function plot_accuracy_and_returns(solver::BetaZeroSolver; expert_accuracy=solver.expert_results.expert_accuracy, expert_returns=solver.expert_results.expert_returns, kwargs...)
    plt_accuracy = plot_accuracy(solver; expert_results=expert_accuracy, expert_label=solver.expert_results.expert_label, kwargs...)
    plt_returns = plot_returns(solver; expert_results=expert_returns, expert_label=solver.expert_results.expert_label, kwargs...)
    return plot(plt_accuracy, plt_returns, layout=2, size=(1000,300), margin=5Plots.mm)
end


"""
"""
function plot_data_gen(solver::BetaZeroSolver;
                       n::Int=10_000, # number of data points to plot
                       xrange=range(0, 5, length=100),
                       yrange=range(-20, 20, length=100),
                       flip_axes::Bool=true,
                       subplots::Bool=true,
                       cmap=nothing,
                       xlabel="\$\\sigma(b)\$",
                       ylabel="\$\\mu(b)\$",
                       title="training data")

    data = sample_data(solver.data_buffer_train, n)

    X1 = flip_axes ? data.X[2,:] : data.X[1,:]
    X2 = flip_axes ? data.X[1,:] : data.X[2,:]
    V = data.Y[1,:]

    if isnothing(cmap)
        cmap_gradient = shifted_colormap(V)
        cmap = [get(cmap_gradient, normalize01(v,V)) for v in V]
    end

    figure = scatter(X1, X2, label=false, xlabel=xlabel, ylabel=ylabel, title=title, cmap=cmap, ms=2, alpha=0.2)
    xlims!(xrange[1], xrange[end])
    ylims!(yrange[1], yrange[end])

    current_xlim = xlims()
    current_ylim = ylims()

    if subplots
        lay = @layout [a{0.3h} _; b{0.7h, 0.7w} c]

        topfig = Plots.histogram(X1, color=:lightgray, linecolor=:gray, normalize=true, label=nothing, xaxis=nothing)
        xlims!(current_xlim) # match limits of main plot
        ylims!(0, ylims()[2]) # anchor at zero

        sidefig = Plots.histogram(X2, color=:lightgray, linecolor=:gray, normalize=true, label=nothing, yaxis=nothing, orientation=:h)
        xlims!(0, xlims()[2]) # anchor at zero
        ylims!(current_ylim) # match limits of main plot
        figure = plot(topfig, figure, sidefig, layout=lay, margin=7Plots.mm)
    end

    return figure
end
