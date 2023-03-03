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
                     expert_results=nothing, # [mean, std]
                     expert_label="expert")

    if metric == :accuracies
        ho_results = [mean_and_std(getfield(m, metric)) for m in solver.holdout_metrics]
        ho_μ = first.(ho_results)
        ho_σ = last.(ho_results)
        pm_x_over_time = accuracy_over_time
    elseif metric == :returns
        ho_μ = [m.mean for m in solver.holdout_metrics]
        ho_σ = [m.std for m in solver.holdout_metrics]
        pm_x_over_time = returns_over_time
    else
        error("`plot_metric` not yet defined for metric $metric")
    end

    n_ho = solver.params.n_holdout
    ho_stderr = ho_σ ./ sqrt(n_ho)

    pm_results = pm_x_over_time(solver, mean_and_std)
    pm_μ = first.(pm_results)
    pm_σ = last.(pm_results)
	n_pm = solver.params.n_data_gen
    pm_stderr = pm_σ ./ sqrt(n_pm)

    plot(title=title, ylabel=ylabel, xlabel=xlabel, legend=:bottomright)

    local pm_X = [1]
    if include_data_gen
        if xaxis_simulations
            pm_X = count_simulations_accumulated(solver; zero_start=true)[1:end-1]
            pm_X = pm_X[1:length(pm_μ)]
        else
            pm_X = eachindex(pm_μ)
        end
        plot!(pm_X, pm_μ, ribbon=pm_stderr, fillalpha=0.2, lw=2, marker=length(pm_μ)==1, label="BetaZero (data collection)", c=:crimson, alpha=0.5)
    end

    local ho_X = [1]
    if include_holdout
        if xaxis_simulations
            ho_X = count_simulations_accumulated(solver; zero_start=true)
            ho_X = ho_X[1:length(ho_μ)]
        else
            ho_X = eachindex(ho_μ)
        end
        plot!(ho_X, ho_μ, ribbon=ho_stderr, fillalpha=0.1, marker=length(ho_μ)==1, lw=1, label="BetaZero (holdout)", c=:darkgreen)
    end

    if !isnothing(expert_results)
        expert_means, expert_stds = expert_results
        hline!([expert_means...], ribbon=expert_stds, fillalpha=0.2, ls=:dash, lw=1, c=:black, label=expert_label)
    end

    if length(pm_μ) != solver.params.n_iterations
        # Currently running, make x-axis span entire iteration domain
        if xaxis_simulations
            xlims!(min(ho_X[1], pm_X[1]), count_simulations(solver))
        else
            xlims!(1, solver.params.n_iterations)
        end
    end

    if metric == :accuracies
        ylims!(ylims()[1], 1.01)
    end

    return plot!()
end

plot_accuracy(solver::BetaZeroSolver; kwargs...) = plot_metric(solver; metric=:accuracies, ylabel="accuracy", kwargs...)
plot_returns(solver::BetaZeroSolver; kwargs...) = plot_metric(solver; metric=:returns, ylabel="returns", kwargs...)
function plot_accuracy_and_returns(solver::BetaZeroSolver; expert_accuracy=solver.expert_results.expert_accuracy, expert_returns=solver.expert_results.expert_returns, kwargs...)
    plt_accuracy = plot_accuracy(solver; kwargs..., expert_results=expert_accuracy, expert_label=solver.expert_results.expert_label)
    plt_returns = plot_returns(solver; kwargs..., expert_results=expert_returns, expert_label=solver.expert_results.expert_label)
    return plot(plt_accuracy, plt_returns, layout=2, size=(1000,300), margin=5Plots.mm)
end
