using MineralExploration
using StatsBase

data_skewness(D) = [skewness(D[x,y,1:end-1]) for x in 1:size(D,1), y in 1:size(D,2)]
data_kurtosis(D) = [kurtosis(D[x,y,1:end-1]) for x in 1:size(D,1), y in 1:size(D,2)]


function convert2data(b::MEBelief)
    states = cat([p.ore_map[:,:,1] for p in particles(b)]..., dims=3)
    observations = zeros(size(states)[1:2])
    for (i,a) in enumerate(b.acts)
        if a.type == :drill
            x, y = a.coords.I
            observations[x,y] = b.obs[i].ore_quality
        end
    end
    return cat(states, observations; dims=3)
end


function BetaZero.input_representation(b::MEBelief)
    D = convert2data(b)
    μ = mean(D[:,:,1:end-1], dims=3)[:,:,1]
    σ² = std(D[:,:,1:end-1], dims=3)[:,:,1]
    sk = data_skewness(D)
    kurt = data_kurtosis(D)
    obs = D[:,:,end]
    return cat(μ, σ², sk, kurt, obs; dims=3)
end
