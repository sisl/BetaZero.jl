# Credit Josh Ott: https://github.com/sisl/SBO_AIPPMS/blob/main/Rover/GP_BMDP_Rover/belief_mdp.jl
"""
    BeliefMDP(pomdp, updater)
Create a belief MDP corresponding to POMDP `pomdp` with belief updates performed by `updater`.
"""
struct BeliefMDP{P<:POMDP, U<:Updater, B, A} <: MDP{B, A}
    pomdp::P
    updater::U
    belief_reward
end

function BeliefMDP(pomdp::P, up::U, belief_reward) where {P<:POMDP, U<:Updater}
    # XXX hack to determine belief type
    b0 = initialize_belief(up, initialstate(pomdp))
    BeliefMDP{P, U, typeof(b0), actiontype(pomdp)}(pomdp, up, belief_reward)
end

function POMDPs.gen(bmdp::BeliefMDP, b, a, rng::AbstractRNG)
    s = rand(rng, b) # NOTE: Different than Josh's implementation
    if isterminal(bmdp.pomdp, s)
        bp = bmdp_handle_terminal(bmdp.pomdp, bmdp.updater, b, s, a, rng::AbstractRNG)::typeof(b)
        return (sp=bp, r=0.0)
    end
    sp, o = @gen(:sp, :o)(bmdp.pomdp, s, a, rng)
    bp = update(bmdp.updater, b, a, o)
    r = bmdp.belief_reward(bmdp.pomdp, b, a, bp)
    return (sp=bp, r=r)
end

POMDPs.actions(bmdp::BeliefMDP{P,U,B,A}, b::B) where {P,U,B,A} = actions(bmdp.pomdp, b)
POMDPs.actions(bmdp::BeliefMDP) = actions(bmdp.pomdp)
POMDPs.actionindex(bmdp::BeliefMDP, a::Int) = findfirst(actions(bmdp) .== a)

POMDPs.isterminal(bmdp::BeliefMDP, b) = all(isterminal(bmdp.pomdp, s) for s in support(b))

POMDPs.discount(bmdp::BeliefMDP) = discount(bmdp.pomdp)

# override this if you want to handle it in a special way
function bmdp_handle_terminal(pomdp::POMDP, updater::Updater, b, s, a, rng)
    @warn("""
         Sampled a terminal state for a BeliefMDP transition - not sure how to proceed, but will try.
         See $(@__FILE__) and implement a new method of POMDPToolbox.bmdp_handle_terminal if you want special behavior in this case.
         """, maxlog=1)
    sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a, rng)
    bp = update(updater, b, a, o)
    return bp
end

function POMDPs.initialstate(bmdp::BeliefMDP)
    return Deterministic(initialize_belief(bmdp.updater, initialstate(bmdp.pomdp)))
end

# deprecated in POMDPs v0.9
function POMDPs.initialstate(bmdp::BeliefMDP, rng::AbstractRNG)
    return initialize_belief(bmdp.updater, initialstate(bmdp.pomdp))
end
