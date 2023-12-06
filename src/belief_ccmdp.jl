"""
    BeliefCCMDP(pomdp, updater)
Create a belief MDP corresponding to POMDP `pomdp` with belief updates performed by `updater`.
"""
struct BeliefCCMDP{P<:POMDP, U<:Updater, B, A} <: MDP{B, A}
    pomdp::P
    updater::U
    belief_reward
    isfailure
end

function BeliefCCMDP(pomdp::P, up::U, belief_reward, isfailure) where {P<:POMDP, U<:Updater}
    # XXX hack to determine belief type
    b0 = initialize_belief(up, initialstate(pomdp))
    BeliefCCMDP{P, U, typeof(b0), actiontype(pomdp)}(pomdp, up, belief_reward, isfailure)
end

function POMDPs.gen(bmdp::BeliefCCMDP, b, a, rng::AbstractRNG)
    s = rand(rng, b) # NOTE: Different than Josh's implementation
    if isterminal(bmdp.pomdp, s)
        bp = bccmdp_handle_terminal(bmdp.pomdp, bmdp.updater, b, s, a, rng::AbstractRNG)::typeof(b)
        return (sp=bp, r=0.0, e=e)
    end
    sp, o = @gen(:sp, :o)(bmdp.pomdp, s, a, rng)
    bp = update(bmdp.updater, b, a, o)
    r = bmdp.belief_reward(bmdp.pomdp, b, a, bp)
    # e = bmdp.isfailure(bmdp.pomdp, s, a) # Check if state is a failure (boolean)
    e = bmdp.isfailure(bmdp.pomdp, b, a) # , bp # Check if belief is a failure (probability)
    return (sp=bp, r=r, e=e)
end

POMDPs.actions(bmdp::BeliefCCMDP{P,U,B,A}, b::B) where {P,U,B,A} = actions(bmdp.pomdp, b)
POMDPs.actions(bmdp::BeliefCCMDP) = actions(bmdp.pomdp)
POMDPs.actionindex(bmdp::BeliefCCMDP, a::Int) = findfirst(actions(bmdp) .== a)

POMDPs.isterminal(bmdp::BeliefCCMDP, b) = all(isterminal(bmdp.pomdp, s) for s in support(b))

POMDPs.discount(bmdp::BeliefCCMDP) = discount(bmdp.pomdp)

# override this if you want to handle it in a special way
function bccmdp_handle_terminal(pomdp::POMDP, updater::Updater, b, s, a, rng)
    @warn("""
         Sampled a terminal state for a BeliefCCMDP transition - not sure how to proceed, but will try.
         See $(@__FILE__) and implement a new method of POMDPToolbox.bccmdp_handle_terminal if you want special behavior in this case.
         """, maxlog=1)
    sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a, rng)
    bp = update(updater, b, a, o)
    return bp
end

function POMDPs.initialstate(bmdp::BeliefCCMDP)
    return Deterministic(initialize_belief(bmdp.updater, initialstate(bmdp.pomdp)))
end

# deprecated in POMDPs v0.9
function POMDPs.initialstate(bmdp::BeliefCCMDP, rng::AbstractRNG)
    return initialize_belief(bmdp.updater, initialstate(bmdp.pomdp))
end
