using Revise
using BetaZero
using ParticleFilters
using POMCPOW
using POMDPs
using POMDPTools
using StatsBase
using MinEx

pomdp = MinExPOMDP()
ds0 = initialstate(pomdp)
up = BootstrapFilter(pomdp, pomdp.N)
b0 = initialize_belief(up, ds0)

# solver = POMCPOWSolver(
#     estimate_value=0.0,
#     criterion=POMCPOW.MaxUCB(1.0),
#     tree_queries=1000,
#     k_action=2.0,
#     alpha_action=0.25,
#     k_observation=2.0,
#     alpha_observation=0.1,
#     tree_in_info=false)

# planner = solve(solver, pomdp)

# for (t,b,a,r) in stepthrough(pomdp, planner, up, "t,b,a,r")
#     @info t, a, r
# end

function simple_minex_accuracy_func(pomdp::POMDP, belief, state, action, returns)
    massive = calc_massive(pomdp, state)
    truth = (massive >= pomdp.extraction_cost) ? :mine : :abandon
    is_correct = (action == truth)
    return is_correct
end

simple_minex_belief_reward = (pomdp::POMDP, b, a, bp)->mean(reward(pomdp, s, a) for s in b.particles)

data_skewness(D) = [skewness(D[x,y,1:end-1]) for x in 1:size(D,1), y in 1:size(D,2)]
data_kurtosis(D) = [kurtosis(D[x,y,1:end-1]) for x in 1:size(D,1), y in 1:size(D,2)]


function convert2data(b::ParticleCollection{MinExState})
    P = b.particles
    grid_dims = size(P[1].ore)
    N = length(P)
    states = Array{Float32}(undef, grid_dims..., N)
    for i in 1:N
        states[:,:,i] = P[i].ore
    end
    # observations = zeros(size(states)[1:2])
    # for (i,a) in enumerate(b.acts)
    #     if a.type == :drill
    #         x, y = a.coords.I
    #         observations[x,y] = b.obs[i].ore_quality
    #     end
    # end
    # return cat(states, observations; dims=3)
    return states
end


function BetaZero.input_representation(b::ParticleCollection{MinExState}; use_higher_orders::Bool=false)
    D = convert2data(b)
    μ = mean(D[:,:,1:end-1], dims=3)[:,:,1]
    σ² = std(D[:,:,1:end-1], dims=3)[:,:,1]
    # obs = D[:,:,end]
    if use_higher_orders
        sk = data_skewness(D)
        kurt = data_kurtosis(D)
        # return cat(μ, σ², sk, kurt, obs; dims=3)
        return cat(μ, σ², sk, kurt; dims=3)
    else
        # return cat(μ, σ², obs; dims=3)
        return cat(μ, σ²; dims=3)
    end
end
