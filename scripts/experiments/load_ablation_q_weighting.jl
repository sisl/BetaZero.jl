ENV["LAUNCH_PARALLEL"] = false
include("../launch_lightdark.jl")

SUFFIXES = ["maxqn", "maxn", "maxq"]
QN_SOLVERS = Dict()
QN_POLICIES = Dict()

for i in eachindex(SUFFIXES)
    suffix = SUFFIXES[i]
    QN_SOLVERS[suffix] = Dict()
    QN_POLICIES[suffix] = Dict()
    for seed in 1:3
        filename_suffix = "lightdark_$(suffix)_seed$seed.bson"
        @info filename_suffix
        solver = BetaZero.load_solver("solver_$filename_suffix")
        policy = BetaZero.load_policy("policy_$filename_suffix")
        QN_SOLVERS[suffix][seed] = solver
        QN_POLICIES[suffix][seed] = policy
        # plot_accuracy_and_returns(solver)
        # Plots.savefig("intermediate_metrics_figure_$filename_suffix.png")
    end
end
