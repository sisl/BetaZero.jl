@with_kw mutable struct OneStepLookaheadSolver <: POMDPs.Solver
    n_actions::Int = 20 # Number of actions to branch.
    n_obs::Int = 5 # Number of observations per action to branch (equal to number of belief updates)
    estimate_value::Function = b->0.0 # Leaf node value estimator
    next_action::Union{Function,Nothing} = nothing # Next action sampler (TODO: Replace with policy head of network)
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
    tried_idxs = []
    tree = Dict()
    A = actions(mdp, s)
    function get_action()
        if isnothing(solver.next_action)
            return rand(A)
        else
            a = solver.next_action(mdp, s, tried_idxs)
            push!(tried_idxs, nothing) # indicate that an action has been taken (see MineralExploration:solver.jl)
            return a
        end
    end

    # for a in branched_actions
    for i in 1:solver.n_actions
        a = get_action()
        if !haskey(tree, a)
            tree[a] = (sp=[], q=[]) # initialize observation set
        end
        for _ in 1:solver.n_obs
            sp, r = @gen(:sp, :r)(mdp, s, a, rng)
            q = r + discount(mdp)*estimate_value(sp)
            push!(tree[a].sp, sp)
            push!(tree[a].q, q)
        end
    end

    # select action based on maximum average Q-value
    best_a = reduce((a,a′) -> mean(tree[a].q) ≥ mean(tree[a′].q) ? a : a′, keys(tree))

    return include_info ? (best_a, tree) : best_a
end


POMDPTools.action_info(planner::OneStepLookaheadPlanner, s) = action(planner, s; include_info=true)