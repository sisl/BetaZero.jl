using BetaZero
using LightDark
using ParticleFilters
using Statistics
using StatsBase
using POMDPs

function BetaZero.input_representation(b::LightDark.ParticleHistoryBelief; use_higher_orders::Bool=false, include_action::Bool=false, include_obs::Bool=false)
    Y = [s.y for s in ParticleFilters.particles(b)]
    μ = mean(Y)
    σ = std(Y)
    local b̃
    if use_higher_orders
        zeroifnan(x) = isnan(x) ? 0 : x
        s = zeroifnan(skewness(Y))
        k = zeroifnan(kurtosis(Y))
        b̃ = Float32[μ, σ, s, k]
    else
        b̃ = Float32[μ, σ]
    end
    if include_obs
        o = isempty(b.observations) ? 0.f0 : b.observations[end]
        b̃ = [b̃..., o]
    end
    if include_action
        a = isempty(b.actions) ? -999 : b.actions[end]
        b̃ = [b̃..., a]
    end
    return b̃
end


# Simpler MLP
function BetaZero.initialize_network(nn_params::BetaZeroNetworkParameters) # MLP
    @info "Using simplified MLP (ReLUs) for neural network..."
    input_size = nn_params.input_size
    action_size = 3 # TODO: add to nn_params.action_size
    bnorm_momentum = 0.6f0
    # p_dropout = 0.5

    # return Chain(
    #     Dense(prod(input_size) => 256, relu),
    #     BatchNorm(256, relu, momentum=bnorm_momentum),
    #     Dense(256 => 256, relu),
    #     BatchNorm(256, relu, momentum=bnorm_momentum),
    #     Dense(256 => 256, relu),
    #     BatchNorm(256, relu, momentum=bnorm_momentum),
    #     Dense(256 => 256, relu),
    #     BatchNorm(256, relu, momentum=bnorm_momentum),
    #     Parallel(vcat,
    #     value_head = Chain(
    #             Dense(256 => 256, relu),
    #             BatchNorm(256, relu, momentum=bnorm_momentum),
    #             Dense(256 => 1),
    #             # Note: A normalization layer will be added during training (with the old layer removed before the next training phase).
    #         ),
    #         policy_head = Chain(
    #             Dense(256 => action_size),
    #             softmax,
    #         )
    #     )
    # )
    return Chain(
        Dense(prod(input_size) => 256, relu),
        BatchNorm(256, relu, momentum=bnorm_momentum),
        Dense(256 => 256, relu),
        BatchNorm(256, relu, momentum=bnorm_momentum),
        Dense(256 => 256, relu),
        BatchNorm(256, relu, momentum=bnorm_momentum),
        Parallel(vcat,
            value_head = Chain(
                Dense(256 => 256, relu),
                BatchNorm(256, relu, momentum=bnorm_momentum),
                Dense(256 => 1),
                # Note: A normalization layer will be added during training (with the old layer removed before the next training phase).
            ),
            policy_head = Chain(
                Dense(256 => 256, relu),
                BatchNorm(256, relu, momentum=bnorm_momentum),
                Dense(256 => action_size),
                softmax,
            )
        )
    )
end

# function BetaZero.initialize_network(nn_params::BetaZeroNetworkParameters) # MLP
#     @info "Using simplified MLP (ReLUs) for neural network..."
#     input_size = nn_params.input_size
#     layer_dims = [256, 256, 256, 256]
#     out_dim = 1
#     bnorm_momentum = 0.6f0
#     p_dropout = 0.5
#     use_dropout = false
#     use_batchnorm = false

#     layers = Any[Dense(prod(input_size), layer_dims[1], relu)]
    
#     for i in 1:length(layer_dims)-1
#         use_dropout && push!(layers, Dropout(p_dropout))
#         use_batchnorm && push!(layers, BatchNorm(layer_dims[i], relu, momentum=bnorm_momentum))
#         push!(layers, Dense(layer_dims[i], layer_dims[i+1], relu))
#     end
#     push!(layers, Dense(layer_dims[end], out_dim))

#     # Note: A normalization layer will be added during training (with the old layer removed before the next training phase).
#     return Chain(layers...)
# end

lightdark_accuracy_func(pomdp, b0, s0, final_action, returns) = returns[end] == pomdp.correct_r
lightdark_belief_reward(pomdp, b, a, bp) = mean(reward(pomdp, s, a) for s in ParticleFilters.particles(b))

POMDPs.convert_s(::Type{A}, b::ParticleHistoryBelief, m::BetaZero.BeliefMDP) where A<:AbstractArray = eltype(A)[BetaZero.input_representation(b)...]
POMDPs.convert_s(::Type{ParticleHistoryBelief}, b::A, m::BetaZero.BeliefMDP) where A<:AbstractArray = ParticleHistoryBelief(particles=ParticleCollection(rand(LDNormalStateDist(b[1], b[2]), 500))) # TODO...
