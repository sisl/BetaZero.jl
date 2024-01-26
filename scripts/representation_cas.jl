using Revise
using BetaZero
using CollisionAvoidancePOMDPs
using Statistics
using StatsBase
using POMDPs

pomdp = CollisionAvoidancePOMDP()
up = CASBeliefUpdater(pomdp)

function BetaZero.input_representation(b::CASBelief)
    μ = collect(Iterators.flatten(mean(b)))
    σ = collect(Iterators.flatten(cov(b)))
    return Float32.(vcat(μ, σ))
end

# TODO: Generalize?
POMDPs.isterminal(bmdp::BeliefMDP, b::CASBelief) = isterminal(bmdp.pomdp, mean(b))
BetaZero.mean_belief_reward(pomdp::POMDP, b::CASBelief, a, bp::CASBelief) = reward(pomdp, mean(b), a)

# executed `stop` while not at the goal, or failed to execute stop at max time (states[end-1] as states = [s0] then pushes sp for every action)
MCTS.isfailure(pomdp::CollisionAvoidancePOMDP, s, a) = CollisionAvoidancePOMDPs.isfailure(pomdp, s)

function MCTS.isfailure(pomdp::CollisionAvoidancePOMDP, b::CASBelief, a)
    # failure probability based on sigma points of the UKF belief
    λ = up.up_ukf.λ
    ukf = b.ukf
    μ, Σ = ukf.μ, ukf.Σ
    w = CollisionAvoidancePOMDPs.weights(μ, λ)
    w = w ./ sum(w) # ensure sums to one (numerical stability of weights)
    S = CollisionAvoidancePOMDPs.sigma_points(μ, Σ, λ)
    F = [MCTS.isfailure(pomdp, bs, a) for bs in S]
    return w'F
end

BetaZero.accuracy(pomdp::CollisionAvoidancePOMDP, b0, s0, states, actions, returns) = !BetaZero.failure(pomdp, b0, s0, states, actions, returns)
BetaZero.failure(pomdp::CollisionAvoidancePOMDP, b0, s0, states, actions, returns) = CollisionAvoidancePOMDPs.isfailure(pomdp, states[end])

function cas_plot_callback(solver, pomdp, up, policy)
    τ = solver.mcts_solver.final_criterion.τ
    policy = online_mode!(solver, policy)
    display(cas_policy_plot(pomdp, up, policy, policy_lookup; use_blur=false))
    H = generate_histories(pomdp, policy, up, 20; parallel=true)
    offline_mode!(solver, policy, τ) # important to reset it with proper τ.
    @show solver.mcts_solver.final_criterion.τ
    return plot_histories(pomdp, H)
end
