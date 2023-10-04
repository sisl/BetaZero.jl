plot_predicted_failure(pomdp::POMDP, policy::BetaZeroPolicy) = plot_predicted_failure(pomdp, policy.surrogate)

function plot_predicted_failure(pomdp::POMDP, surrogate::Surrogate)
    xrange = range(0, 5, length=50)
    yrange = range(-1, 12, length=50)

    plot_config = (size=(500,300), xlabel="belief std", ylabel="belief mean")

    Yv = (x,y)->surrogate(Float32.([x y])')[1]
    Yvdata = [Yv(y,x) for y in yrange, x in xrange]
    cmap_value = BetaZero.shifted_colormap(Yvdata; colors=BetaZero.VALUE_PALETTE)
    plt_value = Plots.heatmap(xrange, yrange, Yvdata; cmap=cmap_value, title="value estimate", plot_config...)

    as = actions(pomdp)
    Yπ = (x,y)->as[argmax(surrogate(Float32.([x y])')[2:end-1])]
    Yπdata = [Yπ(y,x) for y in yrange, x in xrange]
    plt_policy = Plots.heatmap(xrange, yrange, Yπdata; cmap=BetaZero.POLICY_COLOR, levels=2, lw=0, title="policy estimate", plot_config...)

    Ys = (x,y)->surrogate(Float32.([x y])')[end]
    Ysdata = [Ys(y,x) for y in yrange, x in xrange]
    plt_pfail = Plots.contourf(xrange, yrange, Ysdata; cmap=:jet, clims=(0,1), levels=15, lw=0.1, title="predicted probability of failure", plot_config...)
    
    plot(plt_value, plt_policy, plt_pfail, layout=(3,1), size=(400, 700), leftmargin=5Plots.mm)
end
