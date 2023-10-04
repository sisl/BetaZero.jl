using BetaZero
using LightDark
using ParticleBeliefs
using ParticleFilters
using Statistics
using StatsBase
using POMDPs

USE_LIGHTDARK_5 = false

if USE_LIGHTDARK_5
    # Old light dark with light region around 5: LightDark(5)
    pomdp = LightDarkPOMDP(; light_loc=5, sigma = y->abs(y - 5)/sqrt(2) + 1e-2, correct_r=10, incorrect_r=-10)
    @warn "Using old light dark!"
else
    # LightDark(10)
    pomdp = LightDarkPOMDP()
end

up = ParticleHistoryBeliefUpdater(BootstrapFilter(pomdp, 500))

function BetaZero.input_representation(b::ParticleHistoryBelief{LightDarkState};
        include_std::Bool=true, # Important to capture uncertainty in belief.
        use_higher_orders::Bool=false,
        include_action::Bool=false,
        include_obs::Bool=false)
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
        if include_std
            b̃ = Float32[μ, σ]
        else
            b̃ = Float32[μ]
        end
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

BetaZero.accuracy(pomdp::LightDarkPOMDP, b0, s0, states, actions, returns) = returns[end] == pomdp.correct_r
BetaZero.failure(pomdp::LightDarkPOMDP, b0, s0, states, actions, returns) = returns[end] != pomdp.correct_r # either executed `stop` while not at the goal (pomdp.incorrect_r) or did not execute the stop action altogether (return == 0)
# TODO: Change to CPOMDPs.jl cost_func and costs interface.

lightdark_belief_reward(pomdp, b, a, bp) = mean(reward(pomdp, s, a) for s in ParticleFilters.particles(b))

POMDPs.convert_s(::Type{A}, b::ParticleHistoryBelief{LightDarkState}, m::BeliefMDP) where A<:AbstractArray = eltype(A)[BetaZero.input_representation(b)...]
POMDPs.convert_s(::Type{ParticleHistoryBelief{LightDarkState}}, b::A, m::BeliefMDP) where A<:AbstractArray = ParticleHistoryBelief(particles=ParticleCollection(rand(LDNormalStateDist(b[1], b[2]), up.pf.n_init)))


function Statistics.mean(b::ParticleHistoryBelief{LightDarkState})
    return LightDarkState(mean(s->s.status, particles(b)), mean(s->s.y, particles(b)))
end
