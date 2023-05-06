module MinEx
# Credit: Modified from Anthony Corso and John Mern

using POMDPs
using POMDPTools
using Distributions
using Parameters
using ParticleFilters
import MineralExploration: MineralExplorationPOMDP, MEInitStateDist, initialize_data!
using Random
import Base

export
    MinExPOMDP,
    MinExState,
    calc_massive,
    calibrate_extraction_cost,
    extraction_reward,
    MinExStateDistribution,
    MinExHeuristicPolicy


mutable struct MinExState
    ore::Matrix{Float32}
    drill_locations::Vector{Tuple{Int, Int}}
    MinExState(ore, drill_locations=[]) = new(ore, drill_locations)
end


const TERMINAL_LOCATION = (-1,-1)

## Definition of the POMDP
@with_kw struct MinExPOMDP <: POMDP{MinExState, Any, Any}
    grid_dims = (32,32)
    ore_threshold = 0.7
    extraction_cost = 71 # 71 for 32x32 (see `calibrate_extraction_cost`)
    drill_cost = 0.1
    drill_locations = [(i,j) for i=5:5:30 for j=5:5:30]
    terminal_actions = [:abandon, :mine]
    n_particles = 1_000
    σ_abc = 0.1
    γ = 0.999
end


# Ensure MinExState can be compared when adding to a dictionary
Base.hash(s::MinExState, h::UInt) = hash(Tuple(getproperty(s, p) for p in propertynames(s)), h)
Base.isequal(s1::MinExState, s2::MinExState) = all(isequal(getproperty(s1, p), getproperty(s2, p)) for p in propertynames(s1))
Base.:(==)(s1::MinExState, s2::MinExState) = isequal(s1, s2)


# Wrapper
mutable struct MinExStateDistribution
    ds::MEInitStateDist
end

function POMDPs.initialstate(pomdp::MinExPOMDP)
    grid_dims = (pomdp.grid_dims..., 1)
    mineral_pomdp = MineralExplorationPOMDP(grid_dim=grid_dims, high_fidelity_dim=grid_dims)
    initialize_data!(mineral_pomdp, 0)
    return MinExStateDistribution(initialstate_distribution(mineral_pomdp))
end

function Base.rand(rng::Random.AbstractRNG, d::MinExStateDistribution)
    s = rand(rng, d.ds)
    ore = s.ore_map[:,:,1]
    return MinExState(ore)
end
Base.rand(d::MinExStateDistribution) = rand(Random.GLOBAL_RNG, d)
Base.rand(rng::Random.AbstractRNG, d::MinExStateDistribution, n::Int) = [rand(rng, d) for _ in 1:n]
Base.rand(d::MinExStateDistribution, n::Int) = rand(Random.GLOBAL_RNG, d, n)


POMDPs.discount(m::MinExPOMDP) = m.γ
POMDPs.isterminal(m::MinExPOMDP, s) = any(loc->loc == TERMINAL_LOCATION, s.drill_locations)
undrilled_locations(m::MinExPOMDP, b) = undrilled_locations(m::MinExPOMDP, rand(b))
undrilled_locations(m::MinExPOMDP, s::MinExState) = setdiff(m.drill_locations, s.drill_locations)
POMDPs.actions(m::MinExPOMDP, s_or_b) = [m.terminal_actions..., undrilled_locations(m, s_or_b)...]
POMDPs.actions(m::MinExPOMDP) = [m.terminal_actions..., m.drill_locations...]

Base.rand(b::ParticleCollection, n_particles::Int) = [rand(b) for _ in 1:n_particles] # Helper function for sampling multiple states from the posterior
function Base.rand(ds0::Vector{MinExState})
    si = rand(eachindex(ds0))
    s = ds0[si]
    deleteat!(ds0, si) # Remove state from initial state distribution
    return s
end

calc_massive(m::MinExPOMDP, s::MinExState) = sum(s.ore .> m.ore_threshold)
extraction_reward(m::MinExPOMDP, s::MinExState) = calc_massive(m, s) - m.extraction_cost
calibrate_extraction_cost(m::MinExPOMDP, ds0) = mean(calc_massive(m, s) for s in ds0)
# ^ Note: Change extraction cost so state distribution is centered around 0.


# This gen function is for passing multiple drilling actions
function POMDPs.gen(m::MinExPOMDP, s, as::Vector{Tuple{Int, Int}}, rng=Random.GLOBAL_RNG)
    sp = deepcopy(s)
    rtot = 0
    os = Float64[]
    for a in as
        push!(sp.drill_locations, a)
        _, o, r = gen(m, s, a, rng)
        push!(os, o)
        rtot += r
    end
    return (; sp, o=os, r=rtot)
end


function POMDPs.reward(m::MinExPOMDP, s, a)
    if a == :abandon || isterminal(m, s)
        r = 0
    elseif a == :mine
        r = extraction_reward(m, s)
    else
        r = -m.drill_cost
    end
    return r
end


function POMDPs.gen(m::MinExPOMDP, s, a, rng=Random.GLOBAL_RNG)
    # Compute the next state
    sp = deepcopy(s)

    # Compute the reward
    r = reward(m, s, a)
    if r == -m.drill_cost
        push!(sp.drill_locations, a)
    end

    # observation
    if isterminal(m, s) || a in m.terminal_actions
        o = nothing
    else
        o = s.ore[a...]
    end

    if (a in m.terminal_actions || isterminal(m, s))
        push!(sp.drill_locations, TERMINAL_LOCATION)
    end

    return (; sp, o, r)
end


# Function for handling vector of actions (and therefore vector of observations)
function POMDPTools.obs_weight(m::MinExPOMDP, s, a::Vector{Tuple{Int64, Int64}}, sp, o::Vector{Float64})
    w = 1.0
    for (a_i, o_i) in zip(a, o)
        w *= obs_weight(m, s, a_i, sp, o_i)
    end
    return w
end


function POMDPTools.obs_weight(m::MinExPOMDP, s, a, sp, o)
    if (isterminal(m, s) || a in m.terminal_actions)
        w = Float64(isnothing(o))
    else
        w = pdf(Normal(s.ore[a...], m.σ_abc), o)
    end
    return w
end


## Heuristic hand-crafted LCB policy
@with_kw struct MinExHeuristicPolicy <: POMDPs.Policy
    pomdp
    λ = 0.0 # UCB parameter
end


function POMDPs.action(policy::MinExHeuristicPolicy, b)
    pomdp = policy.pomdp
    action_list = pomdp.drill_locations

    legal_actions = actions(pomdp, b)
    for scripted_action in action_list
        ai = findfirst(a->a == scripted_action, legal_actions)
        if !isnothing(ai)
            return scripted_action
        end
    end

    R = [extraction_reward(pomdp, s) for s in particles(b)]
    μ = mean(R)
    σ = std(R)
    lcb = μ - policy.λ*σ

    if lcb > 0
        return :mine
    else
        return :abandon
    end
end


end # module MinEx
