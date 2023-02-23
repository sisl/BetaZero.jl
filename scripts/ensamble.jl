using Flux
using LinearAlgebra
using Parameters
using StatsBase

function lcb(X; λ=1.0)
    μ, σ = mean_and_std(X)
    return μ - λ*σ
end

@with_kw mutable struct EnsambleNetwork
    networks::Vector{Chain} = Chain[]
    weights::Vector{Real} = normalize(ones(length(networks)), 1)
end


function initialize_ensamble(solver, m=3)
    networks = [BetaZero.initialize_network(solver) for _ in 1:m]
    return EnsambleNetwork(networks=networks)
end


function train_ensamble(solver, ensamble::EnsambleNetwork; kwargs...)
    for (i, network) in enumerate(ensamble.networks)
        ensamble.networks[i] = BetaZero.train(network, solver; kwargs...)
    end
    return ensamble
end


function (ensamble::EnsambleNetwork)(x)
    Y = Float32[]
    for network in ensamble.networks
        push!(Y, network(x)[1])
    end
    μ, σ = mean_and_std(Y)
    return μ, σ
end
