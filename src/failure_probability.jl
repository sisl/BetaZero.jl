plot_predicted_failure(pomdp::POMDP, policy::BetaZeroPolicy) = plot_predicted_failure(pomdp, policy.surrogate)

function plot_predicted_failure(pomdp::POMDP, surrogate::Surrogate)
    xrange = range(0, 5, length=50)
    yrange = range(-1, 12, length=50)

    plot_config = (size=(500,300), xlabel="belief std", ylabel="belief mean")

    # t = 0 # pomdp.max_time - 1
    T = 0:10:pomdp.max_time
    # T = pomdp.max_time:pomdp.max_time
    include_time = true
    discrete_actions = true
    max_p_fail = true

    if include_time
        Yv = (x,y)->mean(surrogate(Float32.([x y t])')[1] for t in T)  # last [1] for [y, t]
    else
        Yv = (x,y)->surrogate(Float32.([x y])')[1]
    end
    Yvdata = [Yv(y,x) for y in yrange, x in xrange]
    cmap_value = BetaZero.shifted_colormap(Yvdata; colors=BetaZero.VALUE_PALETTE)
    plt_value = Plots.heatmap(xrange, yrange, Yvdata; cmap=cmap_value, title="value estimate", plot_config...)

    as = actions(pomdp)
    if include_time
        if discrete_actions
            Yπ = (x,y)->as[mode(argmax(surrogate(Float32.([x y t])')[2:end-1]) for t in T)]
            # Yπ = (x,y)->as[round(Int, mean(argmax(surrogate(Float32.([x y t])')[2:end-1]) for t in T))]
        else
            Yπ = (x,y)->mean(as[argmax(surrogate(Float32.([x y t])')[2:end-1])] for t in T)
        end
        levels = 5
    else
        Yπ = (x,y)->as[argmax(surrogate(Float32.([x y])')[2:end-1])]
        levels = 2
    end
    Yπdata = [Yπ(y,x) for y in yrange, x in xrange]
    plt_policy = Plots.heatmap(xrange, yrange, Yπdata; cmap=BetaZero.POLICY_COLOR, levels=levels, lw=0, title="policy estimate", plot_config...)

    if include_time
        if max_p_fail
            Ys = (x,y)->maximum(surrogate(Float32.([x y t])')[end] for t in T)
        else
            Ys = (x,y)->mean(surrogate(Float32.([x y t])')[end] for t in T)
        end
    else
        Ys = (x,y)->surrogate(Float32.([x y])')[end]
    end
    Ysdata = [Ys(y,x) for y in yrange, x in xrange]
    plt_pfail = Plots.contourf(xrange, yrange, Ysdata; cmap=:jet, clims=(0,1), levels=15, lw=0.1, title="predicted probability of failure", plot_config...)
    
    plot(plt_value, plt_policy, plt_pfail, layout=(3,1), size=(400, 700), leftmargin=5Plots.mm)
end
