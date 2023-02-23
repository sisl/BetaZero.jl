function plot_bias(model, data)
    Plots.plot(size=(500,500), ratio=1)
    Plots.xlabel!("true value")
    Plots.ylabel!("predicted value")
    Plots.scatter!(vec(data), vec(model), c=:MediumSeaGreen, label=false, alpha=0.5)
    Plots.abline!(1, 0, label=false, c=:black, lw=2)
end


function policy_plot(policy)


end

function value_plot(policy; σ=0, s=0, k=0, o=0)
    X = -20:0.1:20;
    Y = [policy.network(Float32.([y σ s k o])')[1] for y in X]
    plot(X, Y, label=false, xlabel="belief mean: \$μ(b)\$", ylabel="value est.", c=:crimson, lw=2)
    Plots.hline!([0], label=false, c=:black, style=:dash)
end

function plot_expected_return(solver)

end