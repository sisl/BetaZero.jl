"""
    input_representation(belief)

Interface for belief representation that is input to the neural network.
"""
function input_representation end
input_representation(belief) = error("Please implement `BetaZero.input_representation(belief::YourBeliefType)`")


"""
    accuracy(pomdp::POMDP, b0, s0, states, actions, returns)

Interface for accuracy of agent's decision.
"""
function accuracy end
accuracy(pomdp::POMDP, b0, s0, states, actions, returns) = nothing


"""
    failure(pomdp::POMDP, b0, s0, states, actions, returns)

Interface to determine if agent experienced a failure.
"""
function failure end
failure(pomdp::POMDP, b0, s0, states, actions, returns) = nothing


"""
    optimal_return(pomdp, state)

Interface for computing the optimal return of a POMDP given a `state` (used for relative return calculations).
"""
function optimal_return end
optimal_return(pomdp::POMDP, s) = 0
