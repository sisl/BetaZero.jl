using Revise
using BetaZero
using POMDPs
using CryingBaby

pomdp = CryingBabyPOMDP()
up = updater(pomdp)

# TODO: BetaZero.accuracy(::CryingBabyPOMDP)
function cryingbaby_accuracy_func(pomdp::CryingBabyPOMDP, b0, s0, states, actions, returns)
    accs = []
    for (s,a) in zip(states, actions)
        if s == CryingBaby.hungry
            push!(accs, a == CryingBaby.feed)
        elseif s == CryingBaby.full
            push!(accs, action == CryingBaby.ignore)
        end
    end
    return mean(accs)
end

# TODO: POMDPs.reward(pomdp, b, a, bp)
cryingbaby_belief_reward(pomdp::POMDP, b, a, bp) = reward(pomdp, b, a, bp)
BetaZero.optimal_return(pomdp::CryingBabyPOMDP, s) = 0
function BetaZero.input_representation(b::DiscreteHistoryBelief)
    # Note [p(hungry), p(full)] == [p(hungry), 1 - p(hungry)]
    p_hungry = b.belief.b[1]
    obs = isempty(b.observations) ? -1 : Int(b.observations[end])
    act = isempty(b.actions) ? -1 : Int(b.actions[end])
    # return Float32[p_hungry, act, obs]
    return Float32[p_hungry]
end
