module MinEx
# Credit: Modified from Anthony Corso

using POMDPs
using POMDPTools
using Distributions
using Parameters
using ParticleFilters
using Random
import Base

export
    MinExPOMDP,
    MinExState,
    calc_massive,
    calibrate_extraction_cost

include("utils.jl")

# Helper function for sampling multiple states from the posterior
Base.rand(b::ParticleCollection, N::Int) = [rand(b) for _ in 1:N]

## Definition of the POMDP
@with_kw struct MinExPOMDP <: POMDP{Any, Any, Any}
    ore_threshold = 0.7
    extraction_cost = 71 # 71 for 32x32 (see `calibrate_extraction_cost`)
    drill_cost = 0.1
    drill_locations = [(i,j) for i=3:7:31 for j=3:7:31]
    terminal_actions = [:abandon, :mine]
    N = 1000 # Number of particles
    σ_abc = 0.1
    γ = 0.999
end

mutable struct MinExState
    ore::Matrix{Float32}
    drill_locations::Vector{Tuple{Int, Int}}
    MinExState(ore, drill_locations=[]) = new(ore, drill_locations)
end

# Ensure MinExState can be compared when adding to a dictionary
Base.hash(s::MinExState, h::UInt) = hash(Tuple(getproperty(s, p) for p in propertynames(s)), h)
Base.isequal(s1::MinExState, s2::MinExState) = all(isequal(getproperty(s1, p), getproperty(s2, p)) for p in propertynames(s1))
Base.:(==)(s1::MinExState, s2::MinExState) = isequal(s1, s2)

function POMDPs.initialstate(pomdp::MinExPOMDP; filename="generated_states.h5")
    ore_matrix = load_states(filename)
    indices = shuffle(1:size(ore_matrix,3))[1:pomdp.N]
    return [MinExState(ore_matrix[:,:,i]) for i in indices]
end

POMDPs.initialize_belief(pomdp::MinExPOMDP, ds0) =  ParticleCollection(ds0)
POMDPs.discount(m::MinExPOMDP) = m.γ
POMDPs.isterminal(m::MinExPOMDP, s) = s == :terminal
undrilled_locations(m::MinExPOMDP, b) = undrilled_locations(m::MinExPOMDP, rand(b))
undrilled_locations(m::MinExPOMDP, s::MinExState) = setdiff(m.drill_locations, s.drill_locations)
POMDPs.actions(m::MinExPOMDP, s_or_b) = [m.terminal_actions..., undrilled_locations(m, s_or_b)...]
POMDPs.actions(m::MinExPOMDP) = [m.terminal_actions..., m.drill_locations]

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
    return (;sp, o=os, r=rtot)
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
    sp = (a in m.terminal_actions || isterminal(m, s)) ? :terminal : deepcopy(s)

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

    return (;sp, o, r)
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

## Next action functionality for tree-search solvers
using POMCPOW

struct MinExActionSampler end

# This function is used by POMCPOW to sample a new action for DPW
# In this case, we just want to make sure that we try :mine and :abandon first before drilling
function POMCPOW.next_action(o::MinExActionSampler, problem, b, h)
    # Get the set of children from the current node
    tried_idxs = h.tree isa POMCPOWTree ? h.tree.tried[h.node] : h.tree.children[h.index]
    drill_locations = undrilled_locations(problem, b)
    if length(tried_idxs) == 0 # First visit, try abandon
        return :abandon
    elseif length(drill_locations) == 0 || length(tried_idxs) == 1 # Second visit, try mine
        return :mine
    else # 3+ visit, try drilling
        return rand(drill_locations)
    end
end

end # module MineralX
