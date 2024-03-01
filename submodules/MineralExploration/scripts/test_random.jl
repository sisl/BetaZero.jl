using Revise

using POMDPs
using POMDPSimulators
using POMCPOW
using Plots
using BeliefUpdaters
using Statistics

using MineralExploration

N_INITIAL = 0
MAX_BORES = 10

m = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=2)
initialize_data!(m, N_INITIAL)

ds0 = POMDPs.initialstate_distribution(m)

up = PreviousObservationUpdater()
b0 = MEObservation(nothing, false, false)

solver = RandomSolver()
policy = POMDPs.solve(solver, m)

N = 100
rs = RolloutSimulator(max_steps=25)
V = Float64[]
ores = Float64[]
println("Starting simulations")
for i in 1:N
    if (i%1) == 0
        println("Trial $i")
    end
    s0 = rand(ds0)
    s_massive = s0.ore_map[:,:,1] .>= m.massive_threshold
    r_massive = sum(s_massive)
    v = simulate(rs, m, policy, up, b0, s0)
    push!(V, v)
    push!(ores, r_massive)
end
mean_v = mean(V)
se_v = std(V)/sqrt(N)
println("Discounted Return: $mean_v ± $se_v")

# profitable = ores .> m.extraction_cost
# total_profit = sum(profitable.*(ores .- m.extraction_cost))
# println("Total Profit Available: $total_profit")
#
# mined = (V .> 0.0) .+ (V .< -10.0)
# total_mined = sum(mined)
# println("Total Mined: $total_mined")
# println("Total Abandoned: $(N - total_mined)")
#
# correctly_mined =

profitable_idxs = ores .> m.extraction_cost
loss_idxs = ores .<= m.extraction_cost
n_profitable = sum(profitable_idxs)
n_correct_profit = sum(V[profitable_idxs] .>= 0.0)
p_correct_profit = n_correct_profit/n_profitable

n_loss = sum(loss_idxs)
n_correct_loss = sum(V[loss_idxs] .>= -10.0)
p_correct_loss = n_correct_loss/n_loss

available_profit = sum(ores[profitable_idxs] .- m.extraction_cost)
total_profit = sum(V[profitable_idxs])

println("Profitable: $n_profitable, Drilled: $n_correct_profit, Accuracy: $p_correct_profit")
println("Loss: $n_loss, Abandoned: $n_correct_loss, Accuracy: $p_correct_loss")
println("Available Profit: $available_profit, Total Profit: $total_profit")
