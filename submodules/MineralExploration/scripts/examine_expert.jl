using Revise

using StatsBase
using Plots
using POMDPs
using MineralExploration

function sample_ucb_drill(mean, var)
    scores = belief_scores(mean, var)
    weights = Float64[]
    idxs = CartesianIndex{2}[]
    m, n, _ = size(mean)
    for i =1:m
        for j = 1:n
            idx = CartesianIndex(i, j)
            push!(idxs, idx)
            push!(weights, scores[i, j])
        end
    end
    coords = sample(idxs, StatsBase.Weights(weights))
    # return MEAction(coords=coords)
end

function belief_scores(m, v)

    norm_mean = m[:,:,1]./(maximum(m[:,:,1]) - minimum(m[:,:,1]))
    norm_mean .-= minimum(norm_mean)
    s = sqrt.(v[:,:,1])
    norm_std = s./(maximum(s) - minimum(s)) # actualy using variance
    norm_std .-= minimum(norm_std)
    scores = (norm_mean .* norm_std).^2
    # scores .+= 1.0/(size(m)[1] * size(m)[2])
    # scores = norm_mean .+ 3.0*norm_std
    # scores = norm_mean
    # scores = norm_std
    scores ./= sum(scores)
    return scores
end

N_INITIAL = 0
MAX_BORES = 10
UCB = 50.0
T = 100.0

m = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=2)
initialize_data!(m, N_INITIAL)

ds0 = POMDPs.initialstate_distribution(m)
s0 = rand(ds0)

g = GeoStatsDistribution(m)

up = MEBeliefUpdater(m, g, 1000, 2.0)
println("Initializing belief...")
b0 = POMDPs.initialize_belief(up, ds0)
println("Belief Initialized!")

b = b0

mean_ore, var_ore = MineralExploration.summarize(b)
p_sample = belief_scores(mean_ore, var_ore)
println("Plotting...")
fig = plot(b)
display(fig)
fig = heatmap(p_sample, title="Sampling Probability, UCB=$UCB", fill=true) #, clims=(0.0, 1.0))
display(fig)

x = []
y = []
for _ = 1:100
    coords = sample_ucb_drill(mean_ore, var_ore)
    push!(x, coords[2])
    push!(y, coords[1])
end

scatter!(fig, x, y, legend=:none)
display(fig)

