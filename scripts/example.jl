using Revise
using BetaZero

include("minex_pomdp.jl")
include("minex_representation.jl")

solver = BetaZeroSolver(updater=up)
solver.belief_reward = (pomdp::POMDP, b, a, bp)->mean(reward(pomdp, s, a) for s in particles(b))
solver.mcts_solver.next_action = minexp_next_action # TODO: To be replace with policy head of the network.
solver.network_params.input_size = size(BetaZero.input_representation(b0))
policy = solve(solver, pomdp)
