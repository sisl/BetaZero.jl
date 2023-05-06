using Revise
using BetaZero
using POMDPs
using Tiger

pomdp = TigerPOMDP()
up = updater(pomdp)

# TODO: BetaZero.accuracy(::TigerPOMDP)
function tiger_accuracy_func(pomdp::TigerPOMDP, b0, s0, states, actions, returns)
	return (states[end-1] == Tiger.left && actions[end] == Tiger.open_right) ||
		   (states[end-1] == Tiger.right && actions[end] == Tiger.open_left)
end

# TODO: POMDPs.reward(pomdp, b, a, bp)
tiger_belief_reward(pomdp::POMDP, b, a, bp) = reward(pomdp, b, a, bp)
BetaZero.optimal_return(pomdp::TigerPOMDP, s) = 0
function BetaZero.input_representation(b::DiscreteBelief)
    p_left, p_right, p_done = b.b
    # obs = isempty(b.observations) ? -1 : Int(b.observations[end])
    # act = isempty(b.actions) ? -1 : Int(b.actions[end])
    return Float32[p_left, p_right, p_done]
end

BetaZero.bmdp_handle_terminal(pomdp::TigerPOMDP, updater::Updater, b, s, a, rng) = deepcopy(b)