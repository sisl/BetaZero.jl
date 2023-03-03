"""
Use the raw policy head of the network to get the next action given a belief.
"""
mutable struct RawNetworkPolicy <: Policy
    pomdp::POMDP
    surrogate::Surrogate
end


function POMDPs.action(policy::RawNetworkPolicy, b)
    as = actions(policy.pomdp) # NOTE: actions(pomdp, b) ?
    P = policy_lookup(b, policy.surrogate)
    return as[argmax(P)]
end
