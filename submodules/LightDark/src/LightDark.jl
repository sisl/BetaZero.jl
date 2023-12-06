module LightDark

using Distributions
using Parameters
using POMDPModels
using POMDPTools
using POMDPs
using Random

export
    LightDarkPOMDP,
    LightDarkState,
    LDNormalStateDist

"""
    LightDarkState

## Fields
- `y`: position
- `status`: 0 = normal, negative = terminal
"""
struct LightDarkState
    status::Int64
    y::Float64
    t::Int
end


"""
    LightDarkPOMDP

A one-dimensional light dark problem. The goal is to be near 0. Observations are noisy measurements of the position.
"""
@with_kw mutable struct LightDarkPOMDP <: POMDPs.POMDP{LightDarkState,Int,Float64}
    discount_factor::Float64 = 0.9
    correct_r::Float64 = 100.0
    incorrect_r::Float64 = -100.0
    step_size::Float64 = 1.0
    movement_cost::Float64 = 0.0
    max_y::Float64 = 100.0
    max_time::Int = 99 # 0 -> 99 = 100 steps
    light_loc::Float64 = 10.0
    sigma::Function = y->abs(y - light_loc) + 1e-4
end

POMDPs.discount(p::LightDarkPOMDP) = p.discount_factor
POMDPs.isterminal(::LightDarkPOMDP, act::Int64) = act == 0
POMDPs.isterminal(p::LightDarkPOMDP, s::LightDarkState) = s.status < 0 || s.t > p.max_time
POMDPs.actions(::LightDarkPOMDP) = -1:1

struct LDNormalStateDist
    mean::Float64
    std::Float64
end
struct LDUniformStateDist
    a::Float64
    b::Float64
end

POMDPModels.sampletype(::Type{Union{LDNormalStateDist,LDUniformStateDist}}) = LightDarkState
Base.rand(rng::AbstractRNG, d::LDNormalStateDist) = LightDarkState(0, d.mean + randn(rng)*d.std, 0) # Normally distributed initial state
Base.rand(rng::AbstractRNG, d::LDUniformStateDist) = LightDarkState(0, rand(rng, Distributions.Uniform(d.a, d.b)), 0) # Uniformly distributed initial state
Base.rand(rng::AbstractRNG, d::Union{LDNormalStateDist,LDUniformStateDist}, n::Int) = LightDarkState[rand(rng, d) for _ in 1:n]
Base.rand(d::Union{LDNormalStateDist,LDUniformStateDist}) = rand(Random.GLOBAL_RNG, d)
Base.rand(d::Union{LDNormalStateDist,LDUniformStateDist}, n::Int) = rand(Random.GLOBAL_RNG, d, n)

POMDPs.initialstate(pomdp::LightDarkPOMDP; isuniform::Bool=false) = isuniform ? LDUniformStateDist(-30, 30) : LDNormalStateDist(2, 3)
POMDPs.initialobs(m::LightDarkPOMDP, s) = observation(m, s)

POMDPs.observation(p::LightDarkPOMDP, sp::LightDarkState) = Normal(sp.y, p.sigma(sp.y))

function POMDPs.transition(p::LightDarkPOMDP, s::LightDarkState, a::Int)
    t = s.t + 1
    status = (a == 0 || t > p.max_time) ? -1 : s.status
    y = clamp(s.y + a*p.step_size, -p.max_y, p.max_y)
    return Deterministic(LightDarkState(status, y, t))
end

function POMDPs.reward(p::LightDarkPOMDP, s::LightDarkState, a::Int)
    if s.status < 0
        return 0.0
    elseif a == 0
        if abs(s.y) < 1
            return p.correct_r
        else
            return p.incorrect_r
        end
    else
        return -p.movement_cost
    end
end

# function POMDPs.gen(p::LightDarkPOMDP, s::LightDarkState, a::Int, rng::AbstractRNG=Random.GLOBAL_RNG)
#     sp = rand(transition(p, s, a))
#     r = reward(p, s, a)

POMDPs.convert_s(::Type{A}, s::LightDarkState, p::LightDarkPOMDP) where A<:AbstractArray = eltype(A)[s.status, s.y, s.t]
POMDPs.convert_s(::Type{LightDarkState}, s::A, p::LightDarkPOMDP) where A<:AbstractArray = LightDarkState(Int64(s[1]), s[2], Int64(s[3]))

end # module LightDark
