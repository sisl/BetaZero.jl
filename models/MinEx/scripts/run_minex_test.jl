using Revise
using BetaZero
using MCTS
using ParticleFilters
using Plots; default(fontfamily="Computer Modern", framestyle=:box)
using POMCPOW
using POMDPs
using POMDPTools
using Random
using ParticleBeliefs
using StatsBase
using BSON
using MinEx

Random.seed!(1) # 2 = short and abandon

include(joinpath(@__DIR__, "../../../scripts/representation_minex.jl"))
cache = BSON.load("betazero_policy_2_on_par_pomcpow.bson")[:cache]
params = cache[2].params
nn_params = cache[2].nn_params

solver = BetaZeroSolver(
    pomdp=pomdp,
    updater=up,
    params=params,
    nn_params=nn_params,
    belief_reward=simple_minex_belief_reward,
    accuracy_func=simple_minex_accuracy_func,
)
BetaZero.fill_bmdp!(solver)

network = cache[1]
vhf = network.layers[end].layers.value_head.layers[end]
mean_y = vhf.mean_y.contents
std_y = vhf.std_y.contents
unnormalize_y = y -> (y .* std_y) .+ mean_y
heads = network.layers[end]
value_head = heads.layers.value_head
value_head = Chain(value_head.layers[1:end-1]..., unnormalize_y) # NOTE end-1 to remove it first.
policy_head = heads.layers.policy_head
heads = Parallel(heads.connection, value_head=value_head, policy_head=policy_head)
network = Chain(network.layers[1:end-1]..., heads)


pomcpow_solver = POMCPOWSolver(
    estimate_value=0.0,
    criterion=POMCPOW.MaxUCB(1.0),
    tree_queries=10_000,
    k_action=4.0,
    alpha_action=0.5,
    k_observation=2.0,
    alpha_observation=0.25,
    tree_in_info=false)

pomcpow_planner = solve(pomcpow_solver, pomdp)

# bz_solver = BetaZeroSolver(pomdp=pomdp,
#                            updater=up,
#                             belief_reward=simple_minex_belief_reward,
#                             collect_metrics=true,
#                             verbose=true,
#                             accuracy_func=simple_minex_accuracy_func)


# MCTS parameters
# solver.mcts_solver.n_iterations = 50 # NOTE.
# policy.planner.solver.n_iterations = solver.mcts_solver.n_iterations # NOTE.


# Neural network parameters
# solver.nn_params.use_dirichlet_exploration = true # NOTE!

# Important: resolve/update internal planner parameters
# policy = BetaZero.solve_planner!(solver, policy.surrogate)

# local up = BootstrapFilter(pomdp, pomdp.N)
# up = ParticleHistoryBeliefUpdater(BootstrapFilter(pomdp, pomdp.N))
# solver.updater = up
# solver.bmdp = nothing
# policy = BetaZero.solve_planner!(solver, policy.surrogate)

# f = BetaZero.initialize_network(solver)
# policy = BetaZero.solve_planner!(solver, f)
# f = policy.surrogate

# if !@isdefined(f)
    # f = BetaZero.BSON.load("betazero_policy_1.bson")[:cache][1]
# end

# if !@isdefined(solver)
    # solver = BetaZeroSolver(pomdp=pomdp,
    #                         updater=up,
    #                         belief_reward=simple_minex_belief_reward,
    #                         collect_metrics=true,
    #                         verbose=true,
    #                         plot_incremental_data_gen=true,
    #                         accuracy_func=simple_minex_accuracy_func)

    # solver.mcts_solver.n_iterations = 50 # NOTE.
    # solver.mcts_solver.k_action = 5.0
    # solver.mcts_solver.exploration_constant = 5.0

    # policy = BetaZero.solve_planner!(solver, f)
# end

function extract_mcts(solver, pomdp)
    mcts_solver = deepcopy(solver.mcts_solver)
    mcts_solver.estimate_value = (bmdp,b,d)->0.0
    mcts_solver.next_action = RandomActionGenerator()
    planner = solve(mcts_solver, solver.bmdp)
    return planner
end

# mcts_baseline_planner = extract_mcts(solver, pomdp)

struct MinExAdandonPolicy <: POMDPs.Policy end
POMDPs.action(::MinExAdandonPolicy, b) = :abandon

raw_policy = RawNetworkPolicy(pomdp, network)
value_policy = RawValueNetworkPolicy(solver.bmdp, network, 5)
rand_policy = RandomPolicy(pomdp)
heuristic_policy = MinExHeuristicPolicy(pomdp)
abandon_policy = MinExAdandonPolicy()


policy = BetaZero.solve_planner!(solver, network)

policy2use = value_policy # policy # raw_policy # π_dqn # NOTE

### NOTE.
# POMDPs.action(policy::DiscreteNetwork, b::ParticleHistoryBelief) = action(policy, convert_s(Array, b, mdp))[1]

# max_steps = max(solver.params.max_steps, length(actions(pomdp)))
max_steps = length(actions(pomdp))

global CORRECTS = []
n_sims = 1
@time for i in 1:n_sims
    global policy
    global BELIEFS = []
    global STATES = []
    global ACTIONS = []
    global INFOS = []
    global VOLUMES = []

    local ds0 = initialstate(pomdp)
    local s0 = rand(ds0)
    local b0 = initialize_belief(up, ds0)

    true_extraction_reward = MinEx.extraction_reward(pomdp, s0)
    @info "Immediate extraction reward: $true_extraction_reward"

    @info "Sim $i/$n_sims"
    for (t,b,s,a,r,info) in stepthrough(pomdp, policy2use, up, b0, s0, "t,b,s,a,r,action_info", max_steps=max_steps)
        extraction_rewards = [extraction_reward(pomdp, s) for s in MinEx.particles(b)]
        @info t, a, r, (mean_and_std(extraction_rewards), true_extraction_reward)
        # display(BetaZero.UnicodePlots.histogram(extraction_rewards))

        # P = BetaZero.policy_lookup(policy.surrogate, b)
        # pidx = sortperm(P)
        # BetaZero.UnicodePlots.barplot(actions(pomdp)[pidx], P[pidx]) |> display

        push!(BELIEFS, b)
        push!(STATES, s)
        push!(ACTIONS, a)
        push!(INFOS, info)
        push!(VOLUMES, extraction_rewards)
    end

    local is_correct = simple_minex_accuracy_func(pomdp, nothing, STATES[1], STATES, ACTIONS, nothing)
    @info "Correct? $is_correct"
    println("—"^64)
    push!(CORRECTS, is_correct)
end

if n_sims > 0
    @info "Accuracy: $(mean(CORRECTS)) ± $(std(CORRECTS)/sqrt(length(CORRECTS)))"
end

if false
    Gn = 10
    @time ret_acc_results = [begin
        Random.seed!(i * 50000)
        local ds0 = initialstate(pomdp)
        local s0 = rand(ds0)
        local b0 = initialize_belief(up, ds0)
        history = simulate(HistoryRecorder(max_steps=max_steps), pomdp, policy2use, up, b0, s0)
        G = discounted_reward(history)
        accuracy = solver.accuracy_func(pomdp, history[end].b, history[end].s, history[end].a, G)
        @info i, G, accuracy, extraction_reward(pomdp, s0)
        G, accuracy
    end for i in 1:Gn]
    Gμ, Gσ = mean_and_stderr(first.(ret_acc_results))
    accμ, accσ = mean_and_stderr(last.(ret_acc_results))
    @info "$Gμ ± $(Gσ/sqrt(Gn))"
    @info "$accμ ± $(accσ/sqrt(Gn))"
end

# tikz_policy_plots(pomdp, policy, BELIEFS, ACTIONS; use_mean=true)


#=
begin
    A = collect(keys(INFOS[1][:counts]));
    C = values(INFOS[1][:counts]);
    Q = last.(C);
    qidx = sortperm(Q);
    BetaZero.UnicodePlots.barplot(A[qidx], softmax(Q[qidx]))
end
=#

# Test the gen function with a single action
# s = MinExState(rand(32, 32))
# a = (5, 5)
# sp, o, r = gen(m, s, a)

# @assert sp.ore[a...] == o

# @assert actions(m, sp) == setdiff(actions(m, s), [a])

# obs_weight(m, s, a, sp, o)

# # Test the gen function with multiple actions
# s = MinExState(rand(32, 32))
# as = [(5, 5), (5, 10), (5, 15)]
# sp, os, r = gen(m, s, as, Random.GLOBAL_RNG)
# @assert length(os) == length(as)
# @assert actions(m, sp) == setdiff(actions(m, s), as)

# obs_weight(m, s, a, sp, o)
