using Revise

using POMDPs
using POMDPSimulators
using POMCPOW
using Plots
using ParticleFilters

using POMDPPolicies

using ProfileView

using MineralExploration

N_INITIAL = 0
MAX_BORES = 10

m = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=1)
initialize_data!(m, N_INITIAL)

ds0 = POMDPs.initialstate_distribution(m)
s0 = rand(ds0)

up = MEBeliefUpdater(m, 100)
b0 = POMDPs.initialize_belief(up, ds0)

policy = ExpertPolicy(m)
# @profview POMDPs.action(policy, b0)
# @profview POMDPs.action(policy, b0)

fig = heatmap(s0.ore_map[:,:,1], title="True Ore Field", fill=true, clims=(0.0, 1.0))
# savefig(fig, "./data/example/ore_vals.png")
display(fig)

s_massive = s0.ore_map[:,:,1] .>= 0.7

fig = heatmap(s_massive, title="Massive Ore Deposits", fill=true, clims=(0.0, 1.0))
# savefig(fig, "./data/example/massive.png")
display(fig)

fig = plot(b0, 0)
display(fig)

b_new = nothing
discounted_return = 0.0
println("Entering Simulation...")
for (s, a, r, bp, t) in stepthrough(m, policy, up, b0, s0, "s,a,r,bp,t", max_steps=50)
    global discounted_return
    global b_new
    b_new = bp
    @show t
    @show a
    @show a.coords
    @show r

    @show s.var
    mean_var = MineralExploration.mean_var(bp)
    std_var = MineralExploration.std_var(bp)
    @show mean_var
    @show std_var
    println("==========")
    fig = plot(bp, t)
    str = "./data/example/belief_$t.png"
    # savefig(fig, str)
    display(fig)
    discounted_return += POMDPs.discount(m)^(t - 1)*r
end
println("Episode Return: $discounted_return")
