using Revise

using POMDPs
using Plots

using MineralExploration

N_INITIAL = 0
MAX_BORES = 10

m = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=2)
initialize_data!(m, N_INITIAL)

ds0 = POMDPs.initialstate_distribution(m)
s0 = rand(ds0)

N = 1000
up = MEBeliefUpdater(m, N)
println("Initializing belief...")
# b0 = POMDPs.initialize_belief(up, ds0)
println("Belief Initialized!")

volumes = [sum(b.ore_map[:,:,1] .>= m.massive_threshold) for b in b0.particles.particles]
mean_vol = mean(volumes)
std_vol = MineralExploration.std(volumes)
se_vol = std_vol/sqrt(N)


println("Plotting...")
fig = histogram(volumes, title="Massive Ore Quantities", xaxis="Ore Quantity",
                normalize=:probability, legend=:none)
# fig = heatmap(s0.ore_map[:,:,1], title="True Ore Field", fill=true, clims=(0.0, 1.0))
savefig(fig, "./data/ore_dist.png")
display(fig)

println("Mean Ore Value: $mean_vol ± $se_vol")
println("StdDev Ore Value: $std_vol")

profitable = volumes .> m.extraction_cost
p_profitable = sum(profitable)/N

println("Profitable Fraction: $p_profitable")
