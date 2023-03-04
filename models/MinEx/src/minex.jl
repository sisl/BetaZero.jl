using Revise
using ParticleFilters
using POMCPOW
using POMDPs
using POMDPTools
using MinEx

include("generate_states.jl")

pomdp = MinExPOMDP()
ds0 = initialstate(pomdp)
up = BootstrapFilter(pomdp, pomdp.N)
b0 = initialize_belief(up, ds0)

solver = POMCPOWSolver(
    estimate_value=0.0,
    criterion=POMCPOW.MaxUCB(1.0),
    tree_queries=1000,
    k_action=2.0,
    alpha_action=0.25,
    k_observation=2.0,
    alpha_observation=0.1,
    tree_in_info=false)

planner = solve(solver, pomdp)

for (t,b,a,r) in stepthrough(pomdp, planner, up, "t,b,a,r")
    @info t, a, r
end



# Test the gen function with a single action
# s = MinExState(rand(32, 32))
# a = (5, 5)
# sp, o, r = gen(m, s, a)

# @assert sp.ore[a...] == o

# @assert actions(m, sp) == setdiff(actions(m, s), [a])

# obs_weight(m, s, a, sp, o)

# # Test the gen function with multiple actions
# s = MinExState(rand(32, 32))
# as = [(5, 5), (5, 10), (5, 15)]
# sp, os, r = gen(m, s, as, Random.GLOBAL_RNG)
# @assert length(os) == length(as)
# @assert actions(m, sp) == setdiff(actions(m, s), as)

# obs_weight(m, s, a, sp, o)
