module CryingBaby

using POMDPs
using POMDPTools
using QuickPOMDPs
using Random

export
    CryingBabyPOMDP,
    DiscreteHistoryBelief,
    DiscreteHistoryUpdater

@enum State hungry full
@enum Action feed ignore
@enum Observation crying quiet

struct CryingBabyPOMDP <: POMDP{State, Action, Observation} end

POMDPs.states(pomdp::CryingBabyPOMDP) = [hungry, full]
POMDPs.actions(pomdp::CryingBabyPOMDP) = [feed, ignore]
POMDPs.observations(pomdp::CryingBabyPOMDP) = [crying, quiet]
POMDPs.initialstate(pomdp::CryingBabyPOMDP) = SparseCat(states(pomdp), [0.5, 0.5])
# POMDPs.initialstate(pomdp::CryingBabyPOMDP) = Deterministic(full)
POMDPs.discount(pomdp::CryingBabyPOMDP) = 0.9

POMDPs.stateindex(pomdp::POMDP, s) = findfirst(map(s′->s′ == s, states(pomdp)))
POMDPs.actionindex(pomdp::POMDP, a) = findfirst(map(a′->a′ == a, actions(pomdp)))
POMDPs.obsindex(pomdp::POMDP, o) = findfirst(map(o′->o′ == o, observations(pomdp)))

function POMDPs.transition(pomdp::CryingBabyPOMDP, s::State, a::Action)
    if a == feed
        return SparseCat([hungry, full], [0, 1])
    elseif s == hungry && a == ignore
        return SparseCat([hungry, full], [1, 0])
    elseif s == full && a == ignore
        return SparseCat([hungry, full], [0.1, 0.9])
    end
end

function POMDPs.observation(pomdp::CryingBabyPOMDP, s::State, a::Action, s′::State)
    if s′ == hungry
        return SparseCat([crying, quiet], [0.8, 0.2])
    elseif s′ == full
        return SparseCat([crying, quiet], [0.1, 0.9])
    end
end


POMDPs.reward(pomdp::CryingBabyPOMDP, s::State, a::Action) = (s == hungry ? -10 : 0) + (a == feed ? -5 : 0)


mutable struct DiscreteHistoryBelief
    belief::DiscreteBelief
    observations::Vector
    actions::Vector
end
DiscreteHistoryBelief(belief::DiscreteBelief) = DiscreteHistoryBelief(belief, [], [])


POMDPs.reward(pomdp::CryingBabyPOMDP, b::DiscreteHistoryBelief, a::Action, bp::Union{Nothing,DiscreteBelief}=nothing) = b.belief.b' * [reward(pomdp, s, a) for s in states(pomdp)]


struct DiscreteHistoryUpdater <: Updater
    pomdp::POMDP
    dup::DiscreteUpdater
end
DiscreteHistoryUpdater(pomdp::POMDP) = DiscreteHistoryUpdater(pomdp, DiscreteUpdater(pomdp))

function POMDPs.update(up::DiscreteHistoryUpdater, b::DiscreteHistoryBelief, a, o)
    b′ = update(up.dup, b, a, o)
    observations = push!(deepcopy(b.observations), o)
    actions = push!(deepcopy(b.actions), a)
    return DiscreteHistoryBelief(b′, observations, actions)
end

function POMDPs.initialize_belief(up::DiscreteHistoryUpdater, d)
    belief = initialize_belief(up.dup, d)
    return DiscreteHistoryBelief(belief)
end

Base.rand(rng::AbstractRNG, b::DiscreteHistoryBelief) = rand(rng, b.belief)
POMDPs.support(b::DiscreteHistoryBelief) = support(b.belief)
POMDPs.pdf(b::DiscreteHistoryBelief, s) = pdf(b.belief, s)

Base.hash(s::DiscreteHistoryBelief, h::UInt) = hash(Tuple(getproperty(s, p) for p in propertynames(s)), h)
Base.isequal(s1::DiscreteHistoryBelief, s2::DiscreteHistoryBelief) = all(isequal(getproperty(s1, p), getproperty(s2, p)) for p in propertynames(s1))
Base.:(==)(s1::DiscreteHistoryBelief, s2::DiscreteHistoryBelief) = isequal(s1, s2)

Base.hash(s::DiscreteHistoryUpdater, h::UInt) = hash(Tuple(getproperty(s, p) for p in propertynames(s)), h)
Base.isequal(s1::DiscreteHistoryUpdater, s2::DiscreteHistoryUpdater) = all(isequal(getproperty(s1, p), getproperty(s2, p)) for p in propertynames(s1))
Base.:(==)(s1::DiscreteHistoryUpdater, s2::DiscreteHistoryUpdater) = isequal(s1, s2)

POMDPs.updater(pomdp::CryingBabyPOMDP) = DiscreteHistoryUpdater(pomdp)

end # module CryingBaby

#=
using QMDP
solver = QMDPSolver()
policy = solve(solver, pomdp)

# Query policy for an action, given a belief vector
b = [0.2, 0.8]
a = action(policy, b)
=#
