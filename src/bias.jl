function plot_bias(model, data)
    Plots.plot(size=(500,500), ratio=1)
    Plots.xlabel!("true value")
    Plots.ylabel!("predicted value")
    Plots.scatter!(vec(data), vec(model), c=:MediumSeaGreen, label=false, alpha=0.5)
    Plots.abline!(1, 0, label=false, c=:black, lw=2)
end


function policy_plot()


end