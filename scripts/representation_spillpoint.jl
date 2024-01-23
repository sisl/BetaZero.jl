using Revise
using BetaZero
using SpillpointPOMDP
using ParticleBeliefs
using ParticleFilters
using Statistics
using StatsBase
using POMDPs

exited_reward_amount = 0 # ! NOTE: for `is_constrained` # -1000
exited_reward_binary = 0
# obs_rewards = [-0.1, -0.1]
obs_rewards = [-0.3, -0.7]
height_noise_std = 0.01
topsurface_std = 0.01
sat_noise_std = 0.01
allowable_leakage = 0 # 100 # Ref: Jamgochian, et al. 2022
γ = 0.99

pomdp = SpillpointInjectionPOMDP(; γ, exited_reward_amount, exited_reward_binary, obs_rewards, height_noise_std, topsurface_std, sat_noise_std)
# up = ParticleHistoryBeliefUpdater(BootstrapFilter(pomdp, 20))
# outer_up = SpillpointPOMDP.SIRParticleFilter(
up = SpillpointPOMDP.SIRParticleFilter(
    model=pomdp,
    N=20,
    state2param=SpillpointPOMDP.state2params,
    param2state=SpillpointPOMDP.params2state,
    N_samples_before_resample=10,
    fraction_prior=0.5,
    clampfn=SpillpointPOMDP.clamp_distribution,
    prior=SpillpointPOMDP.param_distribution(initialstate(pomdp)),
    elite_frac=0.3,
    bandwidth_scale=0.5,
    use_threads=true,
    max_cpu_time=0.1 # 1 # 0.5 # 20, 60
)

zeroifnan(x) = isnan(x) ? 0 : x

const SpillpintBelief = Union{ParticleHistoryBelief{SpillpointInjectionState}, SIRParticleBelief{SpillpointInjectionState}, ParticleCollection{SpillpointInjectionState}}
function BetaZero.input_representation(b::SpillpintBelief)
    S = map(s->convert_s(Vector, s, pomdp), ParticleFilters.particles(b))
    μ = collect(Iterators.flatten(mean(S)))
    σ = collect(Iterators.flatten(map(zeroifnan, std(S))))
    return Float32.(vcat(μ, σ))
end

MCTS.isfailure(pomdp::SpillpointInjectionPOMDP, s, a) = s.v_exited > allowable_leakage # allowable exited CO₂
MCTS.isfailure(pomdp::SpillpointInjectionPOMDP, b::SpillpintBelief, a) = mean(MCTS.isfailure(pomdp, s, a) for s in particles(b))

BetaZero.accuracy(pomdp::SpillpointInjectionPOMDP, b0, s0, states, actions, returns) = !BetaZero.failure(pomdp, b0, s0, states, actions, returns)
BetaZero.failure(pomdp::SpillpointInjectionPOMDP, b0, s0, states, actions, returns) = MCTS.isfailure(pomdp, states[end], actions[end])

function POMDPs.actions(m::SpillpointInjectionPOMDP)
    stop_action = (:stop, 0.0)
    drill_actions = [(:drill, val) for val in m.drill_locations]
    injection_actions = [(:inject, val) for val in m.injection_rates]
    observation_actions = [(:observe, config) for config in m.obs_configurations]
	return vcat(stop_action, drill_actions, injection_actions, observation_actions)
end
