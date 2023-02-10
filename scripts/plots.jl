using D3Trees
using Statistics
using StatsBase

showtree(tree) = inchrome(D3Tree(tree))

function BetaZero.MCTS.node_tag(b::MEBelief)
    μ, σ = round.(mean_and_std(MineralExploration.extraction_reward(pomdp, s) for s in particles(b)), digits=4)
    # μ, σ² = MineralExploration.summarize(b)
    # σ = sqrt.(σ²)
    # return "belief: ($(round(mean(μ),digits=4)), $(round(mean(σ),digits=4)))"
    return "belief: ($μ, $σ)"
end

function BetaZero.MCTS.node_tag(a::MEAction)
    if a.type == :drill
        return string("drill at $(a.coords.I)")
    else
        return "$(a.type)"
    end
end
