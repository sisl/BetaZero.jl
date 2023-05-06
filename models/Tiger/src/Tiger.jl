module Tiger

using POMDPs
using POMDPTools
using Parameters

export
    TigerPOMDP

@enum State left right done
@enum Action listen open_left open_right
@enum Observation heard_left heard_right

@with_kw struct TigerPOMDP <: POMDP{State, Action, Observation}
	r_listen = -1
	r_found_tiger = -100
	r_escaped = 10
	p_listen_correctly = 0.85
	discount = 0.95
end

POMDPs.states(pomdp::TigerPOMDP) = [left, right, done]
POMDPs.actions(pomdp::TigerPOMDP) = [listen, open_left, open_right]
POMDPs.observations(pomdp::TigerPOMDP) = [heard_left, heard_right]
POMDPs.initialstate(pomdp::TigerPOMDP) = SparseCat(states(pomdp), [0.5, 0.5, 0.0])
POMDPs.isterminal(pomdp::TigerPOMDP, s::State) = s == done
# POMDPs.isterminal(pomdp::TigerPOMDP, b::DiscreteBelief) = b.b[3] == 1
POMDPs.discount(pomdp::TigerPOMDP) = pomdp.discount

POMDPs.stateindex(pomdp::POMDP, s) = findfirst(map(s′->s′ == s, states(pomdp)))
POMDPs.actionindex(pomdp::POMDP, a) = findfirst(map(a′->a′ == a, actions(pomdp)))
POMDPs.obsindex(pomdp::POMDP, o) = findfirst(map(o′->o′ == o, observations(pomdp)))

function POMDPs.transition(pomdp::TigerPOMDP, s::State, a::Action)
    if a == open_left || a == open_right
		return SparseCat(states(pomdp), [0, 0, 1])
	elseif s == left
		return SparseCat(states(pomdp), [1, 0, 0])
	elseif s == right
		return SparseCat(states(pomdp), [0, 1, 0])
	end
end

function POMDPs.observation(pomdp::TigerPOMDP, a::Action, s::State)
    pc = pomdp.p_listen_correctly
	if a == listen
		p = (s == left) ? pc : 1-pc
        return SparseCat(observations(pomdp), [p, 1-p])
    else
        return SparseCat(observations(pomdp), [0.5, 0.5])
    end
end


function POMDPs.reward(pomdp::TigerPOMDP, s::State, a::Action)
	r = 0.0
	if a == listen
		r += pomdp.r_listen
	elseif a == open_left
		r += (s == left) ? pomdp.r_found_tiger : pomdp.r_escaped
	elseif a == open_right
		r += (s == right) ? pomdp.r_found_tiger : pomdp.r_escaped
	end
	return r
end

POMDPs.reward(pomdp::TigerPOMDP, b::DiscreteBelief, a::Action, bp::Union{Nothing,DiscreteBelief}=nothing) = b.b' * [reward(pomdp, s, a) for s in states(pomdp)]

POMDPs.updater(pomdp::TigerPOMDP) = DiscreteUpdater(pomdp)

end # module Tiger
