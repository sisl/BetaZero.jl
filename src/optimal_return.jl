"""
    optimal_return(pomdp, state)

Interface for computing the optimal return of a POMDP given a `state` (used for relative return calculations).
"""
function optimal_return end
optimal_return(pomdp::POMDP, s) = 0
