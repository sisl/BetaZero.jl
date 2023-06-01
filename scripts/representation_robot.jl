using Revise
using BetaZero
using POMDPs
using ParticleFilters
using Statistics
using RobotLocalization

pomdp = RobotPOMDP()
up = updater(pomdp)

# TODO: BetaZero.accuracy(::RobotPOMDP)
function robot_accuracy_func(pomdp::RobotPOMDP, b0, s0, states, actions, returns)
    return states[end].status == 1 # Correctly left the room
end

zeroifnan(x) = isnan(x) ? 0 : x

# TODO: POMDPs.reward(pomdp, b, a, bp)
robot_belief_reward(pomdp::POMDP, b, a, bp) = reward(pomdp, b, a, bp)
BetaZero.optimal_return(pomdp::RobotPOMDP, s) = 0
function BetaZero.input_representation(b::ParticleCollection{RobotState})
    μ = mean(b)
    σ = std(b)
    return Float32[μ.x, μ.y, μ.theta, zeroifnan(σ.x), zeroifnan(σ.y), zeroifnan(σ.theta)]
end
