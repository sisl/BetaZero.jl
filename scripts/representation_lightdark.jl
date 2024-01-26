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

POMDPs.solve(sol::DESPOTSolver, p::BeliefMDP) = solve(sol, p.pomdp)

zeroifnan(x) = isnan(x) ? 0 : x

function BetaZero.input_representation(b::ScenarioBelief)
    P = collect(particles(b))
    Y = [s.y for s in P]
    t = Float32(P[1].t) # all the same time
    μ = mean(Y)
    σ = zeroifnan(std(Y))
    return Float32[μ, σ]
end

function BetaZero.input_representation(b::ParticleHistoryBelief{LightDarkState};
        include_std::Bool=true, # Important to capture uncertainty in belief.
        use_higher_orders::Bool=false,
        include_action::Bool=false,
        include_obs::Bool=false,
        include_time::Bool=false,
        use_order_invariant_layer=false)
    Y = [s.y for s in ParticleFilters.particles(b)]
    t = Float32(ParticleFilters.particles(b)[1].t) # all the same time
    if use_order_invariant_layer
        return include_time ? vcat(Y, t) : Y
    else
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
        if include_time
            b̃ = vcat(b̃, t)
        end
        return b̃
    end
end

# BetaZero.optimal_return(pomdp::LightDarkPOMDP, s) = pomdp.correct_r

# executed `stop` while not at the goal, or failed to execute stop at max time (states[end-1] as states = [s0] then pushes sp for every action)
function MCTS.isfailure(pomdp::LightDarkPOMDP, s::LightDarkState, a)
    if a == 0
        return abs(s.y) ≥ 1
    else
        return s.t ≥ pomdp.max_time
    end
end

MCTS.isfailure(pomdp::LightDarkPOMDP, b::ParticleHistoryBelief{LightDarkState}, a) = mean(MCTS.isfailure(pomdp, s, a) for s in particles(b))

BetaZero.accuracy(pomdp::LightDarkPOMDP, b0, s0, states, actions, returns) = !BetaZero.failure(pomdp, b0, s0, states, actions, returns)
BetaZero.failure(pomdp::LightDarkPOMDP, b0, s0, states, actions, returns) = MCTS.isfailure(pomdp, states[end], actions[end]) # (states[end-1] as states = [s0] then pushes sp for every action)

lightdark_belief_reward(pomdp, b, a, bp) = mean(reward(pomdp, s, a) for s in ParticleFilters.particles(b))

POMDPs.convert_s(::Type{A}, b::ParticleHistoryBelief{LightDarkState}, m::BeliefMDP) where A<:AbstractArray = eltype(A)[BetaZero.input_representation(b)...]
POMDPs.convert_s(::Type{ParticleHistoryBelief{LightDarkState}}, b::A, m::BeliefMDP) where A<:AbstractArray = ParticleHistoryBelief(particles=ParticleCollection(rand(LDNormalStateDist(b[1], b[2]), up.pf.n_init)))

function Statistics.mean(b::ParticleHistoryBelief{LightDarkState})
    return LightDarkState(mean(s->s.status, particles(b)), mean(s->s.y, particles(b)))
end

function ld_plot_callback(solver, pomdp, up, policy)
    try display(plot_predicted_failure(pomdp, policy.surrogate)) catch end
    psims = gen_lightdark_trajectories(pomdp, up, [policy]; n_sims=10)
    display(plot_policy_trajectories(pomdp, [psims[1]], ["BetaZero"]))
end
