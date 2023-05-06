using BetaZero
using LightDark
using ParticleBeliefs
using ParticleFilters
using Statistics
using StatsBase
using POMDPs

pomdp = LightDarkPOMDP()
up = ParticleHistoryBeliefUpdater(BootstrapFilter(pomdp, 500))

function BetaZero.input_representation(b::ParticleHistoryBelief{LightDarkState}; use_higher_orders::Bool=false, include_action::Bool=false, include_obs::Bool=false)
    Y = [s.y for s in ParticleFilters.particles(b)]
    μ = mean(Y)
    σ = std(Y)
    local b̃
    if use_higher_orders
        zeroifnan(x) = isnan(x) ? 0 : x
        s = zeroifnan(skewness(Y))
        k = zeroifnan(kurtosis(Y))
        b̃ = Float32[μ, σ, s, k]
    else
        b̃ = Float32[μ, σ]
    end
    if include_obs
        o = isempty(b.observations) ? 0.f0 : b.observations[end]
        b̃ = [b̃..., o]
    end
    if include_action
        a = isempty(b.actions) ? -999 : b.actions[end]
        b̃ = [b̃..., a]
    end
    return b̃
end

# BetaZero.optimal_return(pomdp::LightDarkPOMDP, s) = pomdp.correct_r

lightdark_accuracy_func(pomdp, b0, s0, states, actions, returns) = returns[end] == pomdp.correct_r
lightdark_belief_reward(pomdp, b, a, bp) = mean(reward(pomdp, s, a) for s in ParticleFilters.particles(b))

POMDPs.convert_s(::Type{A}, b::ParticleHistoryBelief{LightDarkState}, m::BetaZero.BeliefMDP) where A<:AbstractArray = eltype(A)[BetaZero.input_representation(b)...]
POMDPs.convert_s(::Type{ParticleHistoryBelief{LightDarkState}}, b::A, m::BetaZero.BeliefMDP) where A<:AbstractArray = ParticleHistoryBelief(particles=ParticleCollection(rand(LDNormalStateDist(b[1], b[2]), 500))) # TODO...


function Statistics.mean(b::ParticleHistoryBelief{LightDarkState})
    return LightDarkState(mean(s->s.status, particles(b)), mean(s->s.y, particles(b)))
end
