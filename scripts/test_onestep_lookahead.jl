using Revise
using BetaZero
using NNlib
using Flux
using Plots

include("minex_pomdp.jl")
include("minex_representation.jl")
include("../src/onestep_lookahead.jl")

function Plots.plot(m::MineralExplorationPOMDP, b0::MEBelief, s0::MEState; kwargs...)
    r_massive = calc_massive(m, s0)
    plt_vol = plot_volume(m, b0, r_massive; kwargs...)[1]
    volumes = Float64[calc_massive(m, s) for s in particles(b0)]
    mean_volume = mean(volumes)
    volume_std = std(volumes)
    lcb = mean_volume - volume_std*m.extraction_lcb
    ucb = mean_volume + volume_std*m.extraction_ucb
    vline!([lcb], label=false, c=:black)
    vline!([ucb], label=false, c=:black)
    stop = lcb >= m.extraction_cost || ucb <= m.extraction_cost
    title!("$stop")
    return plt_vol
end

nn_params = BetaZeroNetworkParameters()
random_network = BetaZero.initialize_network(nn_params)
network = BetaZero.load_network(joinpath(@__DIR__, "network_20iters_100gen_200buffer.bson"))
# network = BetaZero.load_network(joinpath(@__DIR__, "network_1iter_100gen_random_retrain.bson"))

solver = OneStepLookaheadSolver(n_actions=5,
                                n_obs=1,
                                estimate_value=b->BetaZero.value_lookup(b, network),
                                # estimate_value=b->BetaZero.value_lookup(b, random_network),
                                next_action=minexp_next_action)
belief_reward = (pomdp::POMDP, b, a, bp)->mean(reward(pomdp, s, a) for s in particles(b))
bmdp = BeliefMDP(pomdp, up, belief_reward)
planner = solve(solver, bmdp)

# @time a = action(planner, b0)
@time data, metrics = BetaZero.run_simulation(pomdp, planner, up, b0, s0; accuracy_func=minex_accuracy_func, collect_metrics=true, include_info=true); metrics.accuracy
