using Flux
using LinearAlgebra
using Parameters
using StatsBase


@with_kw mutable struct EnsambleNetwork
    networks::Vector{Chain} = Chain[]
    weights::Vector{Real} = ones(length(networks))
    mean_only::Bool = true
    λ_lcb::Real = 1.0
end


function initialize_ensamble(solver, m=3)
    networks = [BetaZero.initialize_network(solver) for _ in 1:m]
    return EnsambleNetwork(networks=networks)
end


function train_ensamble(solver, ensamble::EnsambleNetwork; kwargs...)
    for (i, network) in enumerate(ensamble.networks)
        solver.verbose && @info "Training ensamble $i/$(length(ensamble.networks))"
        ensamble.networks[i] = BetaZero.train(network, solver; kwargs...)
    end
    return ensamble
end


function (ensamble::EnsambleNetwork)(x; return_std=false)
    V = Float32[]
    P = Vector{Float32}[]
    for network in ensamble.networks
        y = network(x)
        push!(V, y[1])
        push!(P, y[2:end])
    end
    W = Weights(ensamble.weights)
    Vμ, Vσ = mean_and_std(V, W)
    Pμ, Pσ = mean_and_std(hcat(P...)', W, 1)
    if ensamble.mean_only
        λ = ensamble.λ_lcb
        ṽ = Vμ - λ*Vσ
        p̃ = softmax(Pμ' - λ*Pσ')
        return vcat(ṽ, p̃...)
    else
        μ = vcat(Vμ, Pμ...)
        if return_std
            σ = vcat(Vσ, Pσ...)
            return μ, σ
        else
            return μ
        end
    end
end
