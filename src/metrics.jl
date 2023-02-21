function x_over_time(metrics::Vector, var::Symbol, apply_func::Function=mean; N::Int)
    X = [getfield(m, var) for m in metrics]
    return map(apply_func, collect(Iterators.partition(X, N)))
end

x_over_time(solver::BetaZeroSolver, var::Symbol, apply_func::Function=mean) = x_over_time(solver.performance_metrics, var, apply_func; N=solver.n_data_gen)

accuracy_over_time(solver::BetaZeroSolver, apply_func::Function=mean) = x_over_time(solver, :accuracy, apply_func)
accuracy_over_time(metrics::Vector, apply_func::Function=mean; N) = x_over_time(metrics, :accuracy, apply_func; N=N)
returns_over_time(solver::BetaZeroSolver, apply_func::Function=mean) = x_over_time(solver, :discounted_return, apply_func)
returns_over_time(metrics::Vector, apply_func::Function=mean; N) = x_over_time(metrics, :discounted_return, apply_func; N=N)
num_actions_over_time(solver::BetaZeroSolver, apply_func::Function=mean) = x_over_time(solver, :num_actions, apply_func)
num_actions_over_time(metrics::Vector, apply_func::Function=mean; N) = x_over_time(metrics, :num_actions, apply_func; N=N)
