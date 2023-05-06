using BetaZero
using RockSample
using ParticleFilters
using Statistics
using StatsBase
using POMDPs

pomdp = RockSamplePOMDP()
up = BootstrapFilter(pomdp, 500)

function BetaZero.input_representation(b::ParticleCollection{RSState{3}}; use_higher_orders::Bool=false, include_action::Bool=false, include_obs::Bool=false)
    pos = [s.pos for s in ParticleFilters.particles(b)]
    rocks = [s.rocks for s in ParticleFilters.particles(b)]
    μ_pos = mean(pos)
    σ_pos = std(pos)
    μ_rocks = mean(rocks)
    σ_rocks = std(rocks)

    return Float32[μ_pos..., μ_rocks..., σ_pos..., σ_rocks...]
    # local b̃
    # if use_higher_orders
    #     zeroifnan(x) = isnan(x) ? 0 : x
    #     s = zeroifnan(skewness(Y))
    #     k = zeroifnan(kurtosis(Y))
    #     b̃ = Float32[μ, σ, s, k]
    # else
    #     b̃ = Float32[μ_pos, σ]
    # end
    # if include_obs
    #     o = isempty(b.observations) ? 0.f0 : b.observations[end]
    #     b̃ = [b̃..., o]
    # end
    # if include_action
    #     a = isempty(b.actions) ? -999 : b.actions[end]
    #     b̃ = [b̃..., a]
    # end
    # return b̃
end

function rocksample_accuracy_func(pomdp, b0, s0, states, actions, returns)
    # When we sampled, did we sample only the "good" rocks
    sampled_only_good = all(map(i->begin
        s = states[i]
        a = actions[i]
        if a == RockSample.BASIC_ACTIONS_DICT[:sample] && in(s.pos, pomdp.rocks_positions)
            rock_ind = findfirst(isequal(s.pos), pomdp.rocks_positions)
            s.rocks[rock_ind]
        else
            true
        end
    end, eachindex(actions)))
    # Also ensure that all good rocks were, in fact, sampled (note looking at final state before terminal)
    all_good_rocks_sampled = sum(states[end-1].rocks) == 0
    ended_in_terminal_state = isterminal(pomdp, states[end])
    return sampled_only_good && all_good_rocks_sampled && ended_in_terminal_state
end

rocksample_belief_reward(pomdp, b, a, bp) = mean(reward(pomdp, s, a) for s in ParticleFilters.particles(b))

POMDPs.convert_s(::Type{A}, b::ParticleCollection{RSState{3}}, m::BetaZero.BeliefMDP) where A<:AbstractArray = eltype(A)[BetaZero.input_representation(b)...]
POMDPs.convert_s(::Type{ParticleCollection{RSState{3}}}, b::A, m::BetaZero.BeliefMDP) where A<:AbstractArray = ParticleCollection(particles=ParticleCollection(rand(LDNormalStateDist(b[1], b[2]), 500))) # TODO...


# function Statistics.mean(b::ParticleCollection{RSState{3}})
#     return RSState(mean(s->s.status, particles(b)), mean(s->s.y, particles(b)))
# end
