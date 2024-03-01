using Distributed

#NOTE: DOES NOT RUN WITH GSLIB :(

N_PROCS = 4
println("Building $N_PROCS Workers...")
addprocs(N_PROCS)

using MineralExploration

using POMDPs
using POMDPSimulators
using POMCPOW
using ParticleFilters
using Statistics
using JLD

@everywhere using MineralExploration

@everywhere using POMDPs
@everywhere using POMDPSimulators
@everywhere using POMCPOW
@everywhere using ParticleFilters

N_SIM = 64
N_INITIAL = 0
MAX_BORES = 10

m = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=2)
initialize_data!(m, N_INITIAL)

ds0 = POMDPs.initialstate_distribution(m)

up = MEBeliefUpdater(m, 100)
println("Initializing Belief...")
b0 = POMDPs.initialize_belief(up, ds0)
println("Belief Initialized!")

next_action = NextActionSampler()

solver = POMCPOWSolver(tree_queries=10000,
                       check_repeat_obs=true,
                       check_repeat_act=true,
                       next_action=next_action,
                       k_action=3,
                       alpha_action=0.25,
                       k_observation=2,
                       alpha_observation=0.25,
                       criterion=POMCPOW.MaxUCB(10.0),
                       estimate_value=leaf_estimation
                       )
planner = POMDPs.solve(solver, m)

println("Building Simulation Queue...")
queue = POMDPSimulators.Sim[]
for i = 1:N_SIM
    s0 = rand(ds0)
    s_massive = s0.ore_map[:,:,1] .>= m.massive_threshold
    r_massive = sum(s_massive)
    push!(queue, POMDPSimulators.Sim(m, planner, up, b0, s0, metadata=Dict(:massive_ore=>r_massive)))
end

println("Starting Simulations...")
data = POMDPSimulators.run_parallel(queue, show_progress=true)
println("Simulations Complete!")
JLD.save("./data/POMCPOW_test_4.jld", "results", data)

profitable_idxs = data.massive_ore .> m.extraction_cost
loss_idxs = data.massive_ore .<= m.extraction_cost
n_profitable = sum(profitable_idxs)
n_correct_profit = sum(data.reward[profitable_idxs] .>= 0.0)
p_correct_profit = n_correct_profit/n_profitable

n_loss = sum(loss_idxs)
n_correct_loss = sum(data.reward[loss_idxs] .>= -100.0)
p_correct_loss = n_correct_loss/n_loss

available_profit = sum(data.massive_ore[profitable_idxs] .- m.extraction_cost)
total_profit = sum(data.reward[profitable_idxs])

println("Profitable: $n_profitable, Drilled: $n_correct_profit, Accuracy: $p_correct_profit")
println("Loss: $n_loss, Abandoned: $n_correct_loss, Accuracy: $p_correct_loss")
println("Available Profit: $available_profit, Total Profit: $total_profit")
