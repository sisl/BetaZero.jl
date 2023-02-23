using BetaZero
using LightDark
using ParticleFilters
using Statistics
using StatsBase

function BetaZero.input_representation(b::LightDark.ParticleHistoryBelief; use_higher_orders::Bool=true, include_action::Bool=false)
    Y = [s.y for s in ParticleFilters.particles(b)]
    μ = mean(Y)
    σ = std(Y)
    o = isempty(b.observations) ? 0.f0 : b.observations[end]
    if include_action
        a = isempty(b.actions) ? -999 : b.actions[end]
    end
    if use_higher_orders
        zeroifnan(x) = isnan(x) ? 0 : x
        s = zeroifnan(skewness(Y))
        k = zeroifnan(kurtosis(Y))
        return include_action ? Float32[μ, σ, s, k, a, o] : Float32[μ, σ, s, k, o]
    else
        return include_action ? Float32[μ, σ, a, o] : Float32[μ, σ, o]
    end
end


# Simpler MLP
function BetaZero.initialize_network(nn_params::BetaZeroNetworkParameters) # MLP
    @info "Using simplified MLP (ReLUs) for neural network..."
    input_size = nn_params.input_size
    layer_dims = [256, 256, 256, 256]
    out_dim = 1
    bnorm_momentum = 0.6f0
    use_dropout = true
    p_dropout = 0.75
    use_batchnorm = false

    layers = Any[Dense(prod(input_size), layer_dims[1], relu)]
    
    for i in 1:length(layer_dims)-1
        use_dropout && push!(layers, Dropout(p_dropout))
        use_batchnorm && push!(layers, BatchNorm(layer_dims[i], relu, momentum=bnorm_momentum))
        push!(layers, Dense(layer_dims[i], layer_dims[i+1], relu))
    end
    push!(layers, Dense(layer_dims[end], out_dim))

    # Note: A normalization layer will be added during training (with the old layer removed before the next training phase).
    return Chain(layers...)
end

lightdark_accuracy_func(pomdp, b0, s0, final_action, returns) = returns[end] == pomdp.correct_r
lightdark_belief_reward(pomdp, b, a, bp) = mean(reward(pomdp, s, a) for s in ParticleFilters.particles(b))
