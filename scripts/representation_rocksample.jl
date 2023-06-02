using BetaZero
using RockSample
using ParticleFilters
using Statistics
using LinearAlgebra
using POMDPTools
using StatsBase
using POMDPs
using Plots; default(fontfamily="Computer Modern", framestyle=:box)
Random.seed!(0xC0FFEE)

USE_ROCKSAMPLE_15 = true

if USE_ROCKSAMPLE_15
    n = 15
    k = 15
else
    n = 20
    k = 20
    @warn "RockSample(20): n × n of $n × $n and $k rocks..."
end

pomdp = RockSamplePOMDP((n,n), k)
up = BootstrapFilter(pomdp, 1000)

zeroifnan(x) = isnan(x) ? 0 : x

function BetaZero.input_representation(b::ParticleCollection{RSState{15}}; use_higher_orders::Bool=false, include_action::Bool=false, include_obs::Bool=false)
    pos = ParticleFilters.particles(b)[1].pos # always the same.
    rocks = [s.rocks for s in ParticleFilters.particles(b)]
    μ_rocks = mean(rocks)
    σ_rocks = map(zeroifnan, std(rocks))

    return Float32[pos..., μ_rocks..., σ_rocks...]
end

function BetaZero.accuracy(pomdp::RockSamplePOMDP, b0, s0, states, actions, returns)
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
