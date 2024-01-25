@with_kw mutable struct RockObservations
    ore_quals::Vector{Float64} = Vector{Float64}()
    coordinates::Matrix{Int64} = zeros(Int64, 2, 0)
end

struct MEState{MB}
    ore_map::Array{Float64}  # 3D array of ore_quality values for each grid-cell
    mainbody_params::MB #  Diagonal variance of main ore-body generator
    mainbody_map::Array{Float64}
    rock_obs::RockObservations
    stopped::Bool # Whether or not STOP action has been taken
    decided::Bool # Whether or not the extraction decision has been made
end

function Base.length(obs::RockObservations)
    return length(obs.ore_quals)
end

struct MEObservation
    ore_quality::Union{Float64, Nothing}
    stopped::Bool
    decided::Bool
end

@with_kw struct MEAction
    type::Symbol = :drill
    coords::CartesianIndex = CartesianIndex(0, 0)
end

abstract type GeoDist end

# struct MEBelief{G}
#     particles::Vector{MEState} # Vector of vars & lode maps
#     rock_obs::RockObservations
#     acts::Vector{MEAction}
#     obs::Vector{MEObservation}
#     stopped::Bool
#     decided::Bool
#     geostats::G #GSLIB or GeoStats
# end
