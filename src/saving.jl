"""
Save performance metrics to a file.
"""
function save_metrics(solver::BetaZeroSolver, filename::String)
    metrics = solver.performance_metrics
    BSON.@save filename metrics
end


"""
Save policy to file (MCTS planner and surrogate objects together).
"""
function save_policy(policy::BetaZeroPolicy, filename::String)
    BSON.@save "$filename" policy
end


"""
Load policy from file (MCTS planner and surrogate objects together).
"""
function load_policy(filename::String)
    BSON.@load "$filename" policy
    return policy
end


"""
Save just the surrogate model to a file.
"""
function save_surrogate(policy::BetaZeroPolicy, filename::String)
    surrogate = policy.surrogate
    BSON.@save "$filename" surrogate
end


"""
Load just the surrogate model from a file.
"""
function load_surrogate(filename::String)
    BSON.@load "$filename" surrogate
    return surrogate
end


"""
Save the solver to a file.
"""
function save_solver(solver::BetaZeroSolver, filename::String)
    BSON.@save "$filename" solver
end


"""
Load the solver from a file.
"""
function load_solver(filename::String)
    BSON.@load "$filename" solver
    return solver
end
