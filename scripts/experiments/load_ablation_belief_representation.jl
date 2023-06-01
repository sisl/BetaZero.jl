BR_SOLVERS = Dict()

for suffix in ["mean_only", "maxqn"]
    BR_SOLVERS[suffix] = Dict()
    for seed in 1:3
        filename_suffix = "lightdark_$(suffix)_seed$seed.bson"
        @info filename_suffix
        try
            solver = BetaZero.load_solver("solver_$filename_suffix")
            BR_SOLVERS[suffix][seed] = solver
        catch err
            @warn err
        end
        # plot_accuracy_and_returns(solver)
        # Plots.savefig("intermediate_metrics_figure_$filename_suffix.png")
    end
end
