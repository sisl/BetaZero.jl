using Parameters
using POMDPs
using POMDPTools
using ProgressMeter
using Random
using StatsBase


@with_kw mutable struct OneStepLookaheadSolver <: POMDPs.Solver
    n_actions::Int = 20 # Number of actions to branch.
    n_obs::Int = 5 # Number of observations per action to branch (equal to number of belief updates)
    estimate_value::Function = bp->0.0 # Leaf node value estimator
end


mutable struct OneStepLookaheadPlanner <: POMDPs.Policy
    solver::OneStepLookaheadSolver
    mdp::MDP
    rng::AbstractRNG
end


"""
Greedy one-step lookahead solver. Used as baseline/verifier.
"""
POMDPs.solve(solver::OneStepLookaheadSolver, mdp::MDP) = OneStepLookaheadPlanner(solver, mdp, Random.GLOBAL_RNG)


"""
Return an action from a one-step lookahead planner, based on maximum mean return.
"""
function POMDPs.action(planner::OneStepLookaheadPlanner, s; include_info::Bool=false)
    solver = planner.solver
    mdp = planner.mdp
    estimate_value = solver.estimate_value
    rng = planner.rng
    tree = Dict()
    A = actions(mdp, s)
    branched_actions = sample(A, min(solver.n_actions, length(A)); replace=false) # sample as many actions (without replacement) as requested
    for a in branched_actions
        tree[a] = (sp=[], q=[]) # initialize observation set
        for _ in 1:solver.n_obs
            sp, r = @gen(:sp, :r)(mdp, s, a, rng)
            q = r + discount(mdp)*estimate_value(sp)
            push!(tree[a].sp, sp)
            push!(tree[a].q, q)
        end
    end

    # select action based on maximum average Q-value
    best_a = reduce((a,a′) -> mean(tree[a].q) ≥ mean(tree[a′].q) ? a : a′, keys(tree))

    if include_info
        return best_a, tree
    else
        return best_a
    end
end


POMDPTools.action_info(planner::OneStepLookaheadPlanner, s) = action(planner, s; include_info=true)