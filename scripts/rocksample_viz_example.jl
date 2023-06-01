using POMDPGifs

include("representation_rocksample.jl")
include("rocksample_visualization.jl")

Random.seed!(0xBEEF5)

sim = GifSimulator(filename="rocksample.gif", max_steps=1, fps=5)

## BetaZero
simulate(sim, pomdp, policy, up) # BetaZero

## Random
# simulate(sim, pomdp, RandomPolicy(pomdp), up)

## POMCPOW
# pomcpow_planner = solve_pomcpow(pomdp, nothing; override=true, use_heuristics=false)
# simulate(sim, pomdp, pomcpow_planner, up)

## AdaOPS
# adaops_planner = solve_adaops(pomdp)
# simulate(sim, pomdp, adaops_planner, up)

## Raw Policy Network
# raw_policy = RawNetworkPolicy(pomdp, policy.surrogate)
# simulate(sim, pomdp, raw_policy, up)
