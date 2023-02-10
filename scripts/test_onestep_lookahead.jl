using Revise
using BetaZero

include("minex_pomdp.jl")
include("minex_representation.jl")
include("../src/onestep_lookahead.jl")

nn_params = BetaZeroNetworkParameters()
network = BetaZero.initialize_network(nn_params)
solver = OneStepLookaheadSolver(n_actions=5,
                                n_obs=2,
                                estimate_value=b->BetaZero.value_lookup(b, network),
                                next_action=minexp_next_action)
belief_reward = (pomdp::POMDP, b, a, bp)->mean(reward(pomdp, s, a) for s in particles(b))
bmdp = BeliefMDP(pomdp, up, belief_reward)
planner = solve(solver, bmdp)

# @time a = action(planner, b0)
@time data, metrics = BetaZero.run_simulation(pomdp, planner, up, b0, s0; accuracy_func=minex_accuracy_func, collect_metrics=true); metrics.accuracy
