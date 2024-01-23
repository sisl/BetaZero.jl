function pgfplots_defaults()
    plot(size=(515, 315),
        topmargin=0Plots.mm,
        bottommargin=0Plots.mm,
        leftmargin=0Plots.mm,
        rightmargin=-10Plots.mm,
        titlefont=12, # 40÷2
        legendfontsize=28÷3,
        guidefontsize=28÷3,
        tickfontsize=28÷3,
        colorbartickfontsizes=28÷3,
        framestyle=:box,
        grid=false,
        widen=false,
    )
end


global VALUE_PALETTE = ["#979694", :white, :palegreen4] # Bay green: "#6FA287"
global POLICY_COLOR = :pigeon


"""
Value function heatmaps (when applicable for scalar μ and σ)
"""
function value_plot(policy::BetaZeroPolicy;
                    solver::Union{BetaZeroSolver,Nothing}=nothing,
                    show_data::Bool=true,
                    return_data::Bool=false,
                    return_cmap::Bool=false,
                    n_data::Int=1_000,
                    seed=1000,
                    use_pgf::Bool=false,
                    tight=false,
                    nrange=50,
                    xrange=range(0, 5, length=nrange),
                    yrange=range(-30, 30, length=nrange),
                    restrict_range::Bool=true, # restrict range based on training data
                    flip_axes::Bool=true,
                    xlabel=raw"\(\sigma(b)\)",
                    ylabel=raw"\(\mu(b)\)",
                    title="value",
                    kwargs...)

    Yv = (x,y)->policy.surrogate(Float32.([x y])')[1]

    if show_data
        Random.seed!(seed)
        data_samples = sample_data(solver.data_buffer_train, n_data)

        X1 = flip_axes ? data_samples.X[2,:] : data_samples.X[1,:]
        X2 = flip_axes ? data_samples.X[1,:] : data_samples.X[2,:]
    end

    if restrict_range
        if isnothing(solver)
            @warn("Please include solver as input to value_plot.")
            return nothing
        end
        xmin, xmax = minimum(X1), maximum(X1)
        ymin, ymax = minimum(X2), maximum(X2)

        xrange = range(xmin, xmax, length=nrange)
        yrange = range(ymin, ymax, length=nrange)
    end

    if flip_axes
        # mean on the y-axis, std on the x-axis
        Ydata = [Yv(y,x) for y in yrange, x in xrange] # Note x-y input flip
    else
        # mean on the x-axis, std on the y-axis
        xlabel, ylabel = ylabel, xlabel
        xrange, yrange = yrange, xrange
        Ydata = [Yv(x,y) for y in yrange, x in xrange]
    end
    if isnothing(solver)
        cmap = shifted_colormap(Ydata; colors=VALUE_PALETTE)
    else
        # Include max reward (i.e., max value) when comparing to LAVI.
        cmap = shifted_colormap(vcat(reshape(Ydata, :), solver.pomdp.correct_r); colors=VALUE_PALETTE)
    end

    if use_pgf
        pgfplotsx()
        pgfplots_defaults()
        amsmath = raw"\usepackage{amsmath}"
        if !(amsmath in Plots.PGFPlotsX.DEFAULT_PREAMBLE)
            push!(Plots.PGFPlotsX.DEFAULT_PREAMBLE, amsmath)
        end
    else
        gr()
        plot()
    end

    Plots.heatmap!(xrange, yrange, Ydata, xlabel=xlabel, ylabel=ylabel, title=title, cmap=cmap)
    xl = xlims()
    yl = ylims()

    if show_data
        data = (X1, X2, Ydata)
        scatter!(X1, X2, label=false, c=:black, marker=:circle, msc=:black, ms=0.5, alpha=0.075)
        Plots.annotate!(2.1, 8.0, (raw"\(\leftarrow\) training data distribution", 10, :black, :center))
    end

    plt = plot!(xlims=xl, ylims=yl)
    if tight
        plot!(xlabel="", xticks=false, colorbar=false)
    else
        plot!(rightmargin=10Plots.mm)
    end

    returns = Any[plt]
    if return_data
        push!(returns, data)
    end
    if return_cmap
        push!(returns, cmap)
    end

    if length(returns) == 1
        return returns[1]
    else
        return returns
    end
end


"""
Policy heatmaps (when applicable for scalar μ and σ and discrete actions)
"""
function policy_plot(pomdp::POMDP, policy::BetaZeroPolicy;
                    solver::Union{BetaZeroSolver,Nothing}=nothing,
                    data::Any=nothing,
                    use_pgf::Bool=false,
                    tight=false,
                    nrange=50,
                    xrange=range(0, 5, length=nrange),
                    yrange=range(-30, 30, length=nrange),
                    restrict_range::Bool=true,
                    flip_axes::Bool=true,
                    color=:viridis,
                    xlabel=raw"\(\sigma(b)\)",
                    ylabel=raw"\(\mu(b)\)",
                    title="policy", kwargs...)

    as = actions(pomdp)
    Yπ = (x,y)->as[argmax(policy.surrogate(Float32.([x y])')[2:end-1])] # TODO: find [2:end] code, replace for safety case.

    if restrict_range
        if isnothing(solver)
            @warn("Please include solver as input to value_plot.")
            return nothing
        end

        if isnothing(data)
            ymax, xmax = maximum(solver.data_buffer_train[1].X; dims=2)
            ymin, xmin = minimum(solver.data_buffer_train[1].X; dims=2)
        else
            X1, X2, V = data
            xmin, xmax = minimum(X1), maximum(X1)
            ymin, ymax = minimum(X2), maximum(X2)
        end

        xrange = range(xmin, xmax, length=nrange)
        yrange = range(ymin, ymax, length=nrange)
    end

    if flip_axes
        # mean on the y-axis, std on the x-axis
        Ydata = [Yπ(y,x) for y in yrange, x in xrange] # Note x-y input flip
    else
        # mean on the x-axis, std on the y-axis
        xlabel, ylabel = ylabel, xlabel
        xrange, yrange = yrange, xrange
        Ydata = [Yπ(x,y) for y in yrange, x in xrange]
    end

    if use_pgf
        pgfplotsx()
        pgfplots_defaults()
    else
        gr()
        plot()
    end

    Plots.heatmap!(xrange, yrange, Ydata, xlabel=xlabel, ylabel=ylabel, title=title, cmap=POLICY_COLOR, colorbar=!use_pgf)

    if tight
        plot!(ylabel="", xlabel="", xticks=false, colorbar=false, ytickfont=:white)
    end

    if use_pgf
        Plots.annotate!(1.75, 4, (raw"\(a=\) \texttt up", 10, :black, :center))
        Plots.annotate!(0.5, 3, (raw"\(a=\) \texttt down", 10, :white, :center))
        Plots.annotate!(0.35, 0.0, (raw"\(a=\) \texttt stop", 10, :white, :center))
    end

    return plot!()
end


"""
Plot combined value and policy plots.
"""
function value_and_policy_plot(pomdp::POMDP, policy::BetaZeroPolicy; lavi_policy=nothing, nrange=50, kwargs...)
    include_lavi = !isnothing(lavi_policy)
    use_pgf = kwargs[:use_pgf]
    vplt, data, cmap = value_plot(policy; return_data=true, return_cmap=true, kwargs..., tight=include_lavi, nrange, title="BetaZero raw value network")
    pplt = policy_plot(pomdp, policy; data, kwargs..., tight=include_lavi, nrange, title="BetaZero raw policy network")
    if include_lavi
        # LAVI value plot
        if use_pgf
            pgfplotsx()
            pgfplots_defaults()
        else
            gr()
            plot()
        end
        vplt_lavi = plot!()
        Plots.title!("Approximately optimal value function")
        Plots.xlabel!(raw"\(\sigma(b)\)")
        Plots.ylabel!(raw"\(\mu(b)\)")

        X1, X2, Ydata = data
        xmin, xmax = minimum(X1), maximum(X1)
        ymin, ymax = minimum(X2), maximum(X2)
        xrange = range(xmin, xmax, length=nrange)
        yrange = range(ymin, ymax, length=nrange)
        lavi_Ydata = [value(lavi_policy, [y,x]) for y in yrange, x in xrange] # Note x-y input flip
        # lavi_cmap = cgrad([:white, :green])
        lavi_cmap = shifted_colormap(lavi_Ydata; colors=VALUE_PALETTE)
        heatmap!(xrange, yrange, lavi_Ydata, cmap=lavi_cmap, colorbar=false) # NOTE: x-y flip

        if !use_pgf
            # Independent color bar
            heatmap!(xrange, yrange, Ydata, c=cmap,
                    inset=bbox(0.48, -0.48, 150Plots.px, 200Plots.px, :center),
                    label=false,
                    axis=false,
                    grid=false,
                    bg_inside=nothing,
                    ticks=nothing,
                    subplot=2,
            )
        end

        # LAVI policy plot
        if use_pgf
            pgfplotsx()
            pgfplots_defaults()
        else
            gr()
            plot()
        end
        pplt_lavi = plot!()
        Plots.title!("Approximately optimal policy")
        Plots.xlabel!(raw"\(\sigma(b)\)")

        belief_type = typeof(initialize_belief(lavi_policy.mdp.updater, initialstate(lavi_policy.mdp.pomdp)))
        a2b = (μ,σ)->convert_s(belief_type, [μ,σ], lavi_policy.mdp)
        A = (σ,μ)->action(lavi_policy, a2b(μ,σ)) # NOTE: x-y flip
        heatmap!(xrange, yrange, A, color=POLICY_COLOR, colorbar=false, yticks=false)

        plot(vplt, pplt, vplt_lavi, pplt_lavi, layout=(2,2), size=(1000,400), margin=use_pgf ? 0Plots.mm : 5Plots.mm)
    else
        return plot(vplt, pplt, layout=2, size=(1000,200), margin=use_pgf ? 0Plots.mm : 5Plots.mm)
    end
end


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
                          yrange=range(-30, 30, length=100),
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
                     metric::Symbol=:accuracies, # :accuracies, :returns, :delta
                     xaxis_simulations=false,
                     xlabel=xaxis_simulations ? "total simulations" : "iteration",
                     ylabel="accuracy",
                     title=ylabel,
                     include_holdout=false,
                     include_data_gen=true,
                     relative_to_optimal=false, # TODO: Parameterize
                     apply_rolling_mean=false, # TODO: Parameterize
                     apply_smoothing=true, # TODO: Parameterize
                     ylim_accuracies=false,
                     expert_results=nothing, # [mean, std]
                     expert_label="expert")

    if metric == :accuracies || metric == :delta
        if include_holdout
            ho_results = [mean_and_std(getfield(m, metric)) for m in solver.holdout_metrics]
            ho_μ = first.(ho_results)
            ho_σ = last.(ho_results)
            if apply_rolling_mean
                ho_μ = rolling_mean(ho_μ)
            end
        end
        if metric == :accuracies
            pm_x_over_time = accuracy_over_time
        elseif metric == :delta
            pm_x_over_time = delta_over_time
        end
    elseif metric == :returns
        if include_holdout
            if relative_to_optimal
                ho_μ = [mean([m.returns[i] - m.optimal_returns[i] for i in eachindex(m.returns)]) for m in solver.holdout_metrics]
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
        Plots.hline!([expert_means...], ribbon=expert_stderr, fillalpha=0.2, ls=:dash, lw=1, c=:black, label=expert_label)
    end

    if solver.is_constrained && metric ∈ [:accuracies, :delta]
        if metric == :accuracies
            x_Δ0 = 1 - solver.mcts_solver.Δ0
        elseif metric == :delta
            x_Δ0 = solver.mcts_solver.Δ0
        end
        Plots.hline!([x_Δ0], label=false, c=:gray, ls=:dot)
    end

    # Currently running, make x-axis span entire iteration domain
    if xaxis_simulations
        xlims!(min(ho_X[1], pm_X[1]), count_simulations(solver))
    else
        # TODO: Handle `resume`
        xlims!(1, max(solver.params.n_iterations, length(pm_X)))
    end

    if metric == :accuracies && ylim_accuracies
        ylims!(ylims()[1], 1.01)
    end

    return plot!(legend=metric == :returns ? :bottomright : :right)
end

plot_accuracy(solver::BetaZeroSolver; kwargs...) = plot_metric(solver; metric=:accuracies, ylabel="accuracy", kwargs...)
plot_returns(solver::BetaZeroSolver; kwargs...) = plot_metric(solver; metric=:returns, ylabel="returns", kwargs...)
plot_deltas(solver::BetaZeroSolver; kwargs...) = plot_metric(solver; metric=:delta, ylabel="deltas", kwargs...)
function plot_accuracy_and_returns(solver::BetaZeroSolver; expert_accuracy=solver.expert_results.expert_accuracy, expert_returns=solver.expert_results.expert_returns, kwargs...)
    plt_accuracy = plot_accuracy(solver; expert_results=expert_accuracy, expert_label=solver.expert_results.expert_label, kwargs...)
    plt_returns = plot_returns(solver; expert_results=expert_returns, expert_label=solver.expert_results.expert_label, kwargs...)
    return plot(plt_accuracy, plt_returns, layout=2, size=(1000,300), margin=8Plots.mm, fontfamily="Computer Modern", framestyle=:box)
end
function plot_accuracy_returns_and_deltas(solver::BetaZeroSolver; expert_accuracy=solver.expert_results.expert_accuracy, expert_returns=solver.expert_results.expert_returns, kwargs...)
    plt_accuracy = plot_accuracy(solver; expert_results=expert_accuracy, expert_label=solver.expert_results.expert_label, kwargs...)
    plt_returns = plot_returns(solver; expert_results=expert_returns, expert_label=solver.expert_results.expert_label, kwargs...)
    plt_deltas = plot_deltas(solver; kwargs...)
    return plot(plt_accuracy, plt_returns, plt_deltas, layout=(1,3), size=(1500,300), margin=8Plots.mm, fontfamily="Computer Modern", framestyle=:box)
end


"""
"""
function plot_data_gen(solver::BetaZeroSolver;
                       n::Int=10_000, # number of data points to plot
                       xrange=range(0, 5, length=100),
                       yrange=range(-30, 30, length=100),
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


function plot_ablations(solvers::Dict;
                        metric::Symbol=:returns, # :returns, :accuracies
                        xaxis_simulations::Bool=true,
                        xlabel::String=xaxis_simulations ? "number of episodes" : "iteration",
                        ylabel::String="mean $metric",
                        title::Union{String,Nothing}=nothing,
                        relative_to_optimal::Bool=false,
                        apply_rolling_mean::Bool=false,
                        apply_smoothing::Bool=true,
                        show_raw::Bool=false,
                        show_error_boundary_lines::Bool=true,
                        show_expert::Bool=false,
                        use_pgf::Bool=false,
                        max_iterations=Inf,
                        xticks=5_000:5_000:25_000,
                        yticks=metric == :accuracies ? (0:0.2:1) : (-3:3:12),
                        ablation_type=:q_weighting, # :q_weighting, :action_selection, :output_norm, :belief_rep, :linear_weighting
                    )

    green = "#55a868"
    blue = "#4c72b0"
    light_blue = "#64b5cd"
    red = "#c44e52"
    purple = "#8172b2"
    if ablation_type == :q_weighting
        if isnothing(title)
            title = raw"Ablation: \(Q\)-weighted policy vector"
        end
        criteria = ["maxn", "maxq", "maxqn"]
        criteria_label = Dict(
            "maxqn"=>raw"\(Q\)-weighted counts",
            "maxq"=>raw"\(Q\)-values",
            "maxn"=>"visit counts")
        colors = Dict(
            "maxqn"=>red,
            "maxq"=>purple,
            "maxn"=>light_blue)
    elseif ablation_type == :action_selection
        if isnothing(title)
            title = "Ablation: Prioritized action widening"
        end
        criteria = ["random", "maxqn"]
        criteria_label = Dict(
            "maxqn"=>"Sampled from network",
            "random"=>"Random action")
        colors = Dict(
            "maxqn"=>red,
            "random"=>blue)
    elseif ablation_type == :output_norm
        if isnothing(title)
            title = "Ablation: Output normalization"
        end
        criteria = ["unnormalized", "maxqn"]
        criteria_label = Dict(
            "unnormalized"=>"Unnormalized",
            "maxqn"=>"Normalized")
        colors = Dict(
            "maxqn"=>red,
            "unnormalized"=>purple)
    elseif ablation_type == :belief_rep
        if isnothing(title)
            title = "Ablation: Belief representation"
        end
        criteria = ["mean_only", "maxqn"]
        criteria_label = Dict(
            "mean_only"=>"b = [mean(b)]",
            "maxqn"=>"b = [mean(b), std(b)]")
        colors = Dict(
            "maxqn"=>red,
            "mean_only"=>green)
    elseif ablation_type == :linear_weighting
        if isnothing(title)
            title = raw"Ablation: Linear \(Q\)-weighting"
        end
        criteria = ["maxlinearqn", "maxqn"]
        criteria_label = Dict(
            "maxqn"=>raw"multiplicative combination",
            "maxlinearqn"=>"linear combination")
        colors = Dict(
            "maxqn"=>red,
            "maxlinearqn"=>blue)
    end

    if use_pgf
        pgfplotsx()
    else
        gr()
    end

    plot(title=title,
         ylabel=ylabel,
         xlabel=xlabel,
         legend=:bottomright,
         legend_font_halign=:left,
         size=(515, 315),
         topmargin=5Plots.mm,
         bottommargin=10Plots.mm,
         leftmargin=20Plots.mm,
         rightmargin=10Plots.mm,
         linewidth=2,
         titlefont=15, # 40÷2
         legendfontsize=28÷2,
         guidefontsize=28÷2,
         tickfontsize=28÷2,
         colorbartickfontsizes=28÷2,
         framestyle=:box,
         grid=true,
         widen=false,
         yticks=yticks,
         xticks=(xticks, map(x->x == 0 ? "0" : string(x ÷ 10^3, "K"), xticks)),
    )

    X = Dict()
    Y = Dict()
    Y_err = Dict()
    local expert_results
    local expert_label
    for criterion in keys(solvers)
        local x_result
        y_results = [] # to be averaged across seeds
        y_results_err = [] # to be averaged across seeds

        for seed in keys(solvers[criterion])
            solver = solvers[criterion][seed]

            if metric == :accuracies
                pm_x_over_time = accuracy_over_time
            elseif metric == :returns
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

            pm_results = pm_x_over_time(solver, mean_and_std)
            if max_iterations != Inf
                pm_results = pm_results[1:max_iterations]
            end
            pm_μ = first.(pm_results)
            pm_σ = last.(pm_results)
            n_pm = solver.params.n_data_gen
            pm_stderr = pm_σ ./ sqrt(n_pm)

            if apply_rolling_mean
                pm_μ = rolling_mean(pm_μ)
                pm_stderr = rolling_stderr(pm_μ)
                pm_stderr[1] = 0 # NaN
            end

            local pm_X = [1]

            if xaxis_simulations
                pm_X = count_simulations_accumulated(solver; zero_start=true)[2:end]
                pm_X = pm_X[1:length(pm_μ)]
            else
                pm_X = eachindex(pm_μ)
            end
            x_result = pm_X
            push!(y_results, pm_μ)
            push!(y_results_err, pm_stderr)
            expert_results = solver.expert_results.expert_returns
            expert_label = solver.expert_results.expert_label
        end
        X[criterion] = x_result
        Y[criterion] = y_results
        Y_err[criterion] = y_results_err
    end

    if metric == :accuracies
        ylims!(ylims()[1], 1.01)
    end

    for criterion in criteria
        label = criteria_label[criterion]
        c = colors[criterion]
        x = X[criterion]
        y = mean(Y[criterion])
        # y_err = mean(Y_err[criterion])
        y_err = std(Y[criterion])
        if apply_smoothing
            show_raw && plot!(x, y, alpha=0.3, color=c, label=false)
            plot!(x, smooth(y), color=c, label=label, linewidth=2, fillalpha=0.1, ribbon=smooth(y_err))
            if show_error_boundary_lines
                plot!(x, smooth(y) .+ smooth(y_err), color=c, label=false, linewidth=1, alpha=0.1)
                plot!(x, smooth(y) .- smooth(y_err), color=c, label=false, linewidth=1, alpha=0.1)
            end
        else
            show_raw && plot!(x, y, fillalpha=0.1, lw=2, label=label, c=c, ribbon=y_err)
            if show_error_boundary_lines
                plot!(x, y .+ y_err, color=c, label=false, linewidth=1, alpha=0.1)
                plot!(x, y .- y_err, color=c, label=false, linewidth=1, alpha=0.1)
            end
        end
    end

    if show_expert && !isnothing(expert_results)
        expert_means, expert_stderr = expert_results
        hline!([expert_means...], ribbon=expert_stderr, fillalpha=0.1, ls=:dash, lw=1, c=:black, label=expert_label)
    end

    plt_filename = "plot_ablation_$(ablation_type)"
    if use_pgf
        Plots.savefig(plot!(), "$plt_filename.tex")
    end
    Plots.savefig(plot!(), "$plt_filename.png")

    return nothing
end


function plot_online_performance(results::Dict;
                        xlabel::String="online planning iterations",
                        ylabel::String="mean returns",
                        title::Union{String,Nothing}="Online performance in RockSample(15,15)",
                        apply_smoothing::Bool=false,
                        show_raw::Bool=true,
                        show_error_boundary_lines::Bool=true,
                        use_pgf::Bool=false,
                    )

    green = "#55a868"
    blue = "#4c72b0"
    light_blue = "#64b5cd"
    red = "#c44e52"
    purple = "#8172b2"

    baselines = ["BetaZero", "Raw Network [policy]", "POMCPOW", "Raw Network [value]"] # order

    baseline_labels = Dict(
        "BetaZero"=>"BetaZero",
        "POMCPOW"=>"POMCPOW",
        "AdaOPS"=>"AdaOPS",
        "DESPOT"=>"DESPOT",
        "Raw Network [policy]"=>raw"Raw Policy \(P_\theta\)",
        "Raw Network [value]"=>raw"Raw Value \(V_\theta\)")
    colors = Dict(
        "BetaZero"=>red,
        "POMCPOW"=>blue,
        "AdaOPS"=>:black,
        "DESPOT"=>green,
        "Raw Network [policy]"=>purple,
        "Raw Network [value]"=>green)
    linestyles = Dict(
        "BetaZero"=>:solid,
        "POMCPOW"=>:solid,
        "AdaOPS"=>:dash,
        "DESPOT"=>:dash,
        "Raw Network [policy]"=>:dash,
        "Raw Network [value]"=>:dash)

    if use_pgf
        pgfplotsx()
    else
        gr()
    end

    plot(title=title,
         ylabel=ylabel,
         xlabel=xlabel,
         legend=:topright,
         legend_font_halign=:left,
         size=(515*1.5, 315),
         topmargin=5Plots.mm,
         bottommargin=10Plots.mm,
         leftmargin=20Plots.mm,
         rightmargin=10Plots.mm,
         linewidth=2,
         titlefont=15,
         legendfontsize=28÷2,
         guidefontsize=28÷2,
         tickfontsize=28÷2,
         colorbartickfontsizes=28÷2,
         framestyle=:box,
         grid=true,
         xwiden=false,
         xaxis=:log,
    )

    for k in baselines # keys(results)
        res = results[k]
        label = baseline_labels[k]
        c = colors[k]
        x = res.X
        y = res.Y
        y_err = res.Yerr
        ls = linestyles[k]
        if isempty(x) || length(x) == 1
            # No online iterations, horizontal line
            plot!([10, 10_000_000], [y[1], y[1]], ribbon=[y_err[1], y_err[1]], fillalpha=0.1, ls=ls, lw=1, c=c, label=label)
        else
            if apply_smoothing
                show_raw && plot!(x, y, alpha=0.3, color=c, label=false)
                plot!(x, smooth(y), color=c, ls=ls, mark=true, ms=3, msc=c, mc=:white, label=label, linewidth=2, fillalpha=0.1, ribbon=smooth(y_err))
                if show_error_boundary_lines
                    plot!(x, smooth(y) .+ smooth(y_err), color=c, label=false, linewidth=1, alpha=0.1)
                    plot!(x, smooth(y) .- smooth(y_err), color=c, label=false, linewidth=1, alpha=0.1)
                end
            else
                show_raw && plot!(x, y, ls=ls, fillalpha=0.1, lw=2, label=label, c=c, mark=true, msc=c, mc=:white, ms=3, ribbon=y_err)
                if show_error_boundary_lines
                    plot!(x, y .+ y_err, color=c, label=false, linewidth=1, alpha=0.1)
                    plot!(x, y .- y_err, color=c, label=false, linewidth=1, alpha=0.1)
                end
            end
        end
    end

    plt_filename = "plot_online_performance"
    if use_pgf
        Plots.savefig(plot!(), "$plt_filename.tex")
        gr()
    end
    Plots.savefig(plot!(), "$plt_filename.png")

    return nothing
end


function plot_bias(model, data)
    Plots.plot(size=(500,500), ratio=1)
    Plots.xlabel!("true value")
    Plots.ylabel!("predicted value")
    Plots.scatter!(vec(data), vec(model), c=:MediumSeaGreen, label=false, alpha=0.5)
    Plots.abline!(1, 0, label=false, c=:black, lw=2)
end
