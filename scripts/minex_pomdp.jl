using MineralExploration
using POMCPOW
using POMDPs
using Random

# Random.seed!(5000) # determinism (truth 50x50: ~245 ore volume)
Random.seed!(7) # determinism (truth 30x30: ~230.56 ore volume)

N_INITIAL = 0
MAX_BORES = 25
MIN_BORES = 5
GRID_SPACING = 0
MAX_MOVEMENT = 20

true_grid_dims = (30, 30, 1)
grid_dims = (30, 30, 1)
# true_mainbody = BlobNode(grid_dims=true_grid_dims, factor=4)
true_mainbody = BlobNode(grid_dims=true_grid_dims)
mainbody = BlobNode(grid_dims=grid_dims)

pomdp = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=GRID_SPACING+1, grid_spacing=GRID_SPACING,
                                true_mainbody_gen=true_mainbody, mainbody_gen=mainbody, original_max_movement=MAX_MOVEMENT,
                                min_bores=MIN_BORES, grid_dim=grid_dims, high_fidelity_dim=true_grid_dims,
                                ratio=grid_dims ./ (50,50,1))
initialize_data!(pomdp, N_INITIAL)

ds0 = POMDPs.initialstate_distribution(pomdp)
# s0 = rand(ds0; truth=true) # `truth` is only necessary when doing multi-fidelity MixedFidelityModelSelection analysis and `true_grid_dims` differs from `grid_dims`
s0 = rand(ds0)
up = MEBeliefUpdater(pomdp, 1000, 2.0)
b0 = POMDPs.initialize_belief(up, ds0)

next_action = NextActionSampler()
minexp_next_action(bmdp::BetaZero.BeliefMDP, b::MEBelief, h) = POMCPOW.next_action(next_action, bmdp.pomdp, b, h)

# plot_mass_map(s0.ore_map, pomdp.massive_threshold; dim_scale=pomdp.dim_scale)[1]
