using Revise

using POMDPs
using POMDPSimulators
using POMCPOW
using Plots
using ParticleFilters
using Statistics

using ProfileView
using D3Trees

using MineralExploration

N_INITIAL = 0
MAX_BORES = 50
N_GRID = 3

m = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=2)
initialize_data!(m, N_INITIAL)

ds0 = POMDPs.initialstate_distribution(m)
s0 = rand(ds0)

g = GeoStatsDistribution(m)

up = MEBeliefUpdater(m, g, 100, 2.0)
println("Initializing belief...")
b0 = POMDPs.initialize_belief(up, ds0)
println("Belief Initialized!")

policy = GridPolicy(m, N_GRID, 20)

println("Plotting...")
fig = heatmap(s0.ore_map[:,:,1], title="True Ore Field", fill=true, clims=(0.0, 1.0))
# savefig(fig, "./data/example/ore_vals.png")
display(fig)

s_massive = s0.ore_map[:,:,1] .>= 0.7
r_massive = sum(s_massive)
println("Massive ore: $r_massive")
# println("MB Variance: $(s0.var)")

fig = heatmap(s_massive, title="Massive Ore Deposits: $r_massive", fill=true, clims=(0.0, 1.0))
# savefig(fig, "./data/example/massive.png")
display(fig)

fig = plot(b0)
display(fig)

# vars = [p.var for p in b0.particles]
# mean_vars = mean(vars)
# std_vars = std(vars)
# println("Vars: $mean_vars ± $std_vars")
# fig = histogram(vars, bins=10 )
# display(fig)

b_new = nothing
a_new = nothing
discounted_return = 0.0
B = [b0]
println("Entering Simulation...")
for (sp, a, r, bp, t) in stepthrough(m, policy, up, b0, s0, "sp,a,r,bp,t", max_steps=50)
    global discounted_return
    global b_new
    global a_new
    local fig
    local volumes
    local vars
    local mean_vars
    local std_vars
    a_new = a
    b_new = bp
    @show t
    @show a
    @show r
    @show sp.stopped
    @show bp.stopped

    volumes = Float64[sum(p.ore_map[:,:,1] .>= m.massive_threshold) for p in bp.particles]
    mean_volume = mean(volumes)
    std_volume = std(volumes)
    volume_lcb = mean_volume - 1.0*std_volume
    push!(B, bp)
    @show mean_volume
    @show std_volume
    @show volume_lcb

    fig = plot(bp, t)
    str = "./data/example_grid/belief_$t.png"
    # savefig(fig, str)
    display(fig)

    # vars = [p.var for p in bp.particles]
    # mean_vars = mean(vars)
    # std_vars = std(vars)
    # @show mean_vars
    # @show std_vars
    # fig = histogram(vars, bins=10)
    # display(fig)
    discounted_return += POMDPs.discount(m)^(t - 1)*r
end

fig = plot(b_new)
str = "./data/example_grid/belief_$N_GRID.png"
savefig(fig, str)
# display(fig)

println("Decision: $(a_new.type)")
println("Massive Ore: $r_massive")
println("Mining Profit: $(r_massive - m.extraction_cost)")
println("Episode Return: $discounted_return")
