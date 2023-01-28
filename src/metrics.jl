function x_over_time(solver::BetaZeroSolver, var::Symbol, apply_func::Function=mean)
    X = [getfield(m, var) for m in solver.performance_metrics]
    N = solver.n_data_gen
    return map(apply_func, collect(Iterators.partition(X, N)))
end

function accuracy_over_time(solver::BetaZeroSolver, apply_func::Function=mean)
    return x_over_time(solver, :accuracy, apply_func)
end

function returns_over_time(solver::BetaZeroSolver, apply_func::Function=mean)
    return x_over_time(solver, :discounted_return, apply_func)
end

function num_actions_over_time(solver::BetaZeroSolver, apply_func::Function=mean)
    return x_over_time(solver, :num_actions, apply_func)
end
