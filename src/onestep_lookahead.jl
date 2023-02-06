mutable struct OneStepLookaheadSolver
    n_actions::Int # Number of actions to branch.
    n_obs::Int # Number of observations per action to branch (equal to number of belief updates)
end


mutable struct OneStepLookaheadPlanner{P<:Union{MDP,POMDP}}
    solver::OneStepLookaheadSolver
    problem::P
end


"""
Greedy one-step lookahead solver. Used as baseline/verifier.
"""
POMDPs.solve(solver::OneStepLookaheadSolver, problem::Union{MDP,POMDP}) = OneStepLookaheadPlanner(solver, problem)


"""
Return an action from a one-step lookahead planner, based on maximum mean return.
"""
function POMDPs.action(planner::OneStepLookaheadPlanner, s)

end
