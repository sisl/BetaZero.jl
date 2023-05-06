using Revise
using POMDPs
using POMDPTools
using Plots; default(fontfamily="Computer Modern", framestyle=:box)
using Random
using MCTS # dev GuMCTS
using D3Trees
using StatsBase
using BetaZero
using LightDark
using ParticleBeliefs
using LinearAlgebra
using Flux
include("representation_lightdark.jl")
include("plot_lightdark.jl")
include("tree_visualization.jl")

function MCTS.node_tag(b)
    b̃ = BetaZero.input_representation(b)
    rd = x->round(x, digits=2)
    return "belief: $(rd(b̃[1])) ± $(rd(b̃[2]))"
end

function MCTS.node_tag(a::Int)
    return "$a"
end


pomdp = LightDarkPOMDP()
up = ParticleHistoryBeliefUpdater(BootstrapFilter(pomdp, 500))


using BetaZero.Flux, BetaZero.Flux.NNlib, BetaZero.Flux.MLUtils, BetaZero.MCTS, BetaZero.GaussianProcesses, BetaZero.DataStructures, BetaZero.Random

# betazero_solver = BetaZeroSolver(pomdp=pomdp, updater=up)
# network = BetaZero.initialize_network(betazero_solver)

bz_policy = BetaZero.load_policy("data/policy_lightdark_pre_gumbel_10.bson")
network = bz_policy.surrogate
# network = policy.surrogate

enum_solver = 2
if enum_solver == 1
    @info "DPWSolver"
    solver = DPWSolver(
        n_iterations=100,
        tree_in_info=true,
        # estimate_value=(bmdp,b,d)->0,
        enable_action_pw=false,
        exploration_constant=1.0,
        k_action=2.0,
        alpha_action=0.25,
        k_state=2.0,
        alpha_state=0.1,
        estimate_value=(bmdp,b,d)->BetaZero.value_lookup(network, b),
    )
elseif enum_solver == 2
    @info "GumbelSolver"
    solver = GumbelSolver(
        n_iterations=100,
        tree_in_info=true,
        exploration_constant=1.0,
        k_action=2.0,
        alpha_action=0.25,
        k_state=2.0,
        alpha_state=0.1,
        # estimate_value=(bmdp,b,d)->0,
        # estimate_value=(bmdp,b,d)->randn(),
        # estimate_value=(bmdp,b,d)->mean(reward(pomdp, s, :mine) for s in particles(b)),
        estimate_value=(bmdp,b,d)->BetaZero.value_lookup(network, b),
        estimate_policy=(bmdp,b)->BetaZero.policy_lookup(network, b),
        # estimate_policy=(bmdp,b)->normalize(rand(3), 1),
        # estimate_policy=(bmdp,b)->[0.33, 0.33, 0.33],
        # depth=5,
        )
elseif enum_solver == 3
    @info "DARSolver"
    nn_params = BetaZeroNetworkParameters(input_size=BetaZero.get_input_size(pomdp,up), action_size=length(actions(pomdp)))
    solver = DARSolver(n_iterations=100,
        exploration_constant=1.0,
        k_action=2.0,
        alpha_action=0.25,
        k_state=2.0,
        alpha_state=0.1,
        estimate_value=(bmdp,b,d)->BetaZero.value_lookup(network, b),
        estimate_policy=(bmdp,b)->BetaZero.policy_lookup(network, b),
        next_action=(bmdp,b,bnode)->BetaZero.next_action(bmdp, b, network, nn_params, bnode),
        tree_in_info=true,
        counts_in_info=true)
end
bmdp = BeliefMDP(pomdp, up, lightdark_belief_reward)
planner = solve(solver, bmdp)

returns = []
accuracies = []
all_trees = []
all_steps = []
for i in 0:0
    Random.seed!(i)
    println("—"^40)
    @info "Seed $i"
    local steps = []
    local max_steps = 50
    local trees = []
    for (t,s,a,r,sp,b,o,bp,ai) in stepthrough(pomdp, planner, up, "t,s,a,r,sp,b,o,bp,action_info"; max_steps=max_steps)
        # @info "Simulation time $t/$max_steps: $a ($(BetaZero.input_representation(b))) $(s.y)"
        # if all(map(_s->isterminal(pomdp, _s), support(bp)))
        #     @warn "Reached terminal belief (all state particles terminal)"
        #     break
        # end
        push!(steps, (;s,a,r,sp,b,o,bp))
        push!(trees, ai[:tree])
    end
    push!(all_trees, trees)
    push!(all_steps, steps)
    
    R = map(step->step.r, steps)
    ret = BetaZero.compute_returns(R; γ=discount(pomdp))
    push!(returns, ret[1])

    correct = lightdark_accuracy_func(pomdp, nothing, nothing, nothing, nothing, ret)
    @info "Correct? $correct"
    push!(accuracies, correct)
end

@info "Returns = $(mean_and_std(returns)), Accuracy = $(mean_and_std(accuracies))"
