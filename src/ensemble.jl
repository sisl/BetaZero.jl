using Flux
using LinearAlgebra
using Parameters
using StatsBase


@with_kw mutable struct EnsembleNetwork
    networks::Vector{Chain} = Chain[]
    weights::Vector{Real} = ones(length(networks))
    mean_only::Bool = true
    λ_lcb::Real = 1.0
end


function initialize_ensemble(solver::BetaZeroSolver, m::Int=3; kwargs...)
    networks = [BetaZero.initialize_network(solver) for _ in 1:m]
    return EnsembleNetwork(; networks, kwargs...)
end


function train(ensemble::EnsembleNetwork, solver::BetaZeroSolver; kwargs...)
    for (i, network) in enumerate(ensemble.networks)
        solver.verbose && @info "Training ensemble $i/$(length(ensemble.networks))"
        ensemble.networks[i] = BetaZero.train(network, solver; kwargs...)
    end
    return ensemble
end


function (ensemble::EnsembleNetwork)(x; return_std=false)
    V = Float32[]
    P = Vector{Float32}[]
    for network in ensemble.networks
        y = network(x)
        push!(V, y[1])
        push!(P, y[2:end])
    end
    W = Weights(ensemble.weights)
    Vμ, Vσ = mean_and_std(V, W)
    Pμ, Pσ = mean_and_std(hcat(P...)', W, 1)
    if ensemble.mean_only
        μ = vcat(Vμ, Pμ...)
        if return_std
            σ = vcat(Vσ, Pσ...)
            return μ, σ
        else
            return μ
        end
    else # LCB
        λ = ensemble.λ_lcb
        ṽ = Vμ - λ*Vσ
        p̃ = softmax(Pμ' - λ*Pσ')
        return vcat(ṽ, p̃...)
    end
end
