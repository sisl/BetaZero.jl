module ParticleBeliefs

using Reexport
using Parameters
using Random
@reexport using ParticleFilters
@reexport using POMDPs

export
    ParticleHistoryBelief,
    ParticleHistoryBeliefUpdater

@with_kw mutable struct ParticleHistoryBelief{T}
    particles::Union{ParticleCollection{T}, Vector{T}}
    observations::Vector = []
    actions::Vector = []
end

@with_kw mutable struct ParticleHistoryBeliefUpdater <: Updater
    pf::Updater
end

function POMDPs.update(up::ParticleHistoryBeliefUpdater, b::ParticleHistoryBelief, a, o)
    particles = update(up.pf, b.particles, a, o)
    observations = push!(deepcopy(b.observations), o)
    actions = push!(deepcopy(b.actions), a)
    return ParticleHistoryBelief(particles, observations, actions)
end

function POMDPs.update(up::ParticleHistoryBeliefUpdater, b::ParticleCollection, a, o)
    particles = update(up.pf, b, a, o)
    observations = [o]
    actions = [a]
    return ParticleHistoryBelief(particles, observations, actions)
end

function POMDPs.initialize_belief(up::ParticleHistoryBeliefUpdater, d)
    particles = initialize_belief(up.pf, d)
    return ParticleHistoryBelief(; particles)
end

function Base.rand(rng::AbstractRNG, b::ParticleHistoryBelief, n::Integer=1)
    if n == 1
        return rand(rng, particles(b))
    else
        return rand(rng, particles(b), n)
    end
end

ParticleFilters.particles(b::ParticleHistoryBelief) = particles(b.particles)
ParticleFilters.support(b::ParticleHistoryBelief) = particles(b)
ParticleFilters.pdf(b::ParticleHistoryBelief, s) = pdf(b.particles, s)

Base.hash(s::ParticleHistoryBelief, h::UInt) = hash(Tuple(getproperty(s, p) for p in propertynames(s)), h)
Base.isequal(s1::ParticleHistoryBelief, s2::ParticleHistoryBelief) = all(isequal(getproperty(s1, p), getproperty(s2, p)) for p in propertynames(s1))
Base.:(==)(s1::ParticleHistoryBelief, s2::ParticleHistoryBelief) = isequal(s1, s2)

Base.hash(s::ParticleCollection, h::UInt) = hash(Tuple(getproperty(s, p) for p in propertynames(s)), h)
Base.isequal(s1::ParticleCollection, s2::ParticleCollection) = all(isequal(getproperty(s1, p), getproperty(s2, p)) for p in propertynames(s1))
Base.:(==)(s1::ParticleCollection, s2::ParticleCollection) = isequal(s1, s2)

end # module ParticleBeliefs
