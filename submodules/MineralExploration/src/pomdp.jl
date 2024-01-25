
@with_kw struct MineralExplorationPOMDP <: POMDP{MEState, MEAction, MEObservation}
    reservoir_dims::Tuple{Float64, Float64, Float64} = (2000.0, 2000.0, 30.0) #  lat x lon x thick in meters
    grid_dim::Tuple{Int64, Int64, Int64} = (50, 50, 1) #  dim x dim grid size
    max_bores::Int64 = 10 # Maximum number of bores
    min_bores::Int64 = 1 # Minimum number of bores
    max_movement::Int64 = 0 # Maximum distanace between bores. 0 denotes no restrictions
    initial_data::RockObservations = RockObservations() # Initial rock observations
    delta::Int64 = 1 # Minimum distance between wells (grid coordinates)
    grid_spacing::Int64 = 0 # Number of cells in between each cell in which wells can be placed
    drill_cost::Float64 = 0.1
    strike_reward::Float64 = 1.0
    extraction_cost::Float64 = 150.0
    extraction_lcb::Float64 = 0.1
    extraction_ucb::Float64 = 0.1
    variogram::Tuple = (0.005, 30.0, 0.0001) #sill, range, nugget
    # nugget::Tuple = (1, 0)
    geodist_type::Type = GeoStatsDistribution # GeoDist type for geo noise
    gp_mean::Float64 = 0.25
    mainbody_weight::Float64 = 0.6
    mainbody_gen::MainbodyGen = SingleFixedNode(grid_dims=grid_dim)
    massive_threshold::Float64 = 0.7
    rng::AbstractRNG = Random.GLOBAL_RNG
end

function GeoStatsDistribution(p::MineralExplorationPOMDP)
    variogram = SphericalVariogram(sill=p.variogram[1], range=p.variogram[2],
                                    nugget=p.variogram[3])
    domain = CartesianGrid{Int64}(p.grid_dim[1], p.grid_dim[2])
    return GeoStatsDistribution(grid_dims=p.grid_dim,
                                data=deepcopy(p.initial_data),
                                domain=domain,
                                mean=p.gp_mean,
                                variogram=variogram)
end

function GSLIBDistribution(p::MineralExplorationPOMDP)
    variogram = (1, 1, 0.0, 0.0, 0.0, p.variogram[2], p.variogram[2], 1.0)
    # variogram::Tuple = (1, 1, 0.0, 0.0, 0.0, 30.0, 30.0, 1.0)
    return GSLIBDistribution(grid_dims=p.grid_dim, n=p.grid_dim,
                            data=deepcopy(p.initial_data), mean=p.gp_mean,
                            sill=p.variogram[1], variogram=variogram,
                            nugget=p.variogram[3])
end

"""
    sample_coords(dims::Tuple{Int, Int}, n::Int)
Sample coordinates from a Cartesian grid of dimensions given by dims and return
them in an array
"""
function sample_coords(dims::Tuple{Int, Int, Int}, n::Int)
    idxs = CartesianIndices(dims)
    samples = sample(idxs, n)
    sample_array = Array{Int64}(undef, 2, n)
    for (i, sample) in enumerate(samples)
        sample_array[1, i] = sample[1]
        sample_array[2, i] = sample[2]
    end
    return (samples, sample_array)
end

function sample_initial(p::MineralExplorationPOMDP, n::Integer)
    coords, coords_array = sample_coords(p.grid_dim, n)
    dist = GeoStatsDistribution(p)
    state = rand(dist)
    ore_quality = state[coords]
    return RockObservations(ore_quality, coords_array)
end

function sample_initial(p::MineralExplorationPOMDP, coords::Array)
    n = length(coords)
    coords_array = Array{Int64}(undef, 2, n)
    for (i, sample) in enumerate(coords)
        coords_array[1, i] = sample[1]
        coords_array[2, i] = sample[2]
    end
    dist = GeoStatsDistribution(p)
    state = rand(dist)
    ore_quality = state[coords]
    return RockObservations(ore_quality, coords_array)
end

function initialize_data!(p::MineralExplorationPOMDP, n::Integer)
    new_rock_obs = sample_initial(p, n)
    append!(p.initial_data.ore_quals, new_rock_obs.ore_quals)
    p.initial_data.coordinates = hcat(p.initial_data.coordinates, new_rock_obs.coordinates)
    return p
end

function initialize_data!(p::MineralExplorationPOMDP, coords::Array)
    new_rock_obs = sample_initial(p, coords)
    append!(p.initial_data.ore_quals, new_rock_obs.ore_quals)
    p.initial_data.coordinates = hcat(p.initial_data.coordinates, new_rock_obs.coordinates)
    return p
end

POMDPs.discount(::MineralExplorationPOMDP) = 0.99
POMDPs.isterminal(m::MineralExplorationPOMDP, s::MEState) = s.decided

struct MEInitStateDist
    gp_distribution::GeoDist
    mainbody_weight::Float64
    mainbody_gen::MainbodyGen
    massive_thresh::Float64
    rng::AbstractRNG
end

function POMDPs.initialstate(m::MineralExplorationPOMDP)
    gp_dist = m.geodist_type(m)
    MEInitStateDist(gp_dist, m.mainbody_weight, m.mainbody_gen,
                    m.massive_threshold, m.rng)
end

function Base.rand(d::MEInitStateDist, n::Int=1)
    gp_ore_maps = Base.rand(d.rng, d.gp_distribution, n)
    if n == 1
        gp_ore_maps = Array{Float64, 3}[gp_ore_maps]
    end

    states = MEState[]
    x_dim = d.gp_distribution.grid_dims[1]
    y_dim = d.gp_distribution.grid_dims[2]
    for i = 1:n
        lode_map, lode_params = rand(d.rng, d.mainbody_gen)
        max_lode = maximum(lode_map)
        lode_map ./= max_lode
        lode_map .*= d.mainbody_weight
        lode_map = repeat(lode_map, outer=(1, 1, 1))

        gp_ore_map = gp_ore_maps[i]
        ore_map = lode_map + gp_ore_map
        state = MEState(ore_map, lode_params, lode_map,
                RockObservations(), false, false)
        push!(states, state)
    end
    if n == 1
        return states[1]
    else
        return states
    end
end

Base.rand(rng::AbstractRNG, d::MEInitStateDist, n::Int=1) = rand(d, n)

function extraction_reward(m::MineralExplorationPOMDP, s::MEState)
    s_massive = s.ore_map .>= m.massive_threshold
    r = m.strike_reward*sum(s_massive)
    r -= m.extraction_cost
    return r
end

function POMDPs.gen(m::MineralExplorationPOMDP, s::MEState, a::MEAction, rng::Random.AbstractRNG)
    if a ∉ POMDPs.actions(m, s)
        error("Invalid Action $a from state $s")
    end
    stopped = s.stopped
    decided = s.decided
    a_type = a.type
    if a_type == :stop && !stopped && !decided
        obs = MEObservation(nothing, true, false)
        r = 0.0
        rock_obs_p = s.rock_obs
        stopped_p = true
        decided_p = false
    elseif a_type == :abandon && stopped && !decided
        obs = MEObservation(nothing, true, true)
        r = 0.0
        rock_obs_p = s.rock_obs
        stopped_p = true
        decided_p = true
    elseif a_type == :mine && stopped && !decided
        obs = MEObservation(nothing, true, true)
        r = extraction_reward(m, s)
        rock_obs_p = s.rock_obs
        stopped_p = true
        decided_p = true
    elseif a_type ==:drill && !stopped && !decided
        ore_obs = s.ore_map[a.coords[1], a.coords[2], 1]
        a = reshape(Int64[a.coords[1] a.coords[2]], 2, 1)
        r = -m.drill_cost
        rock_obs_p = deepcopy(s.rock_obs)
        rock_obs_p.coordinates = hcat(rock_obs_p.coordinates, a)
        push!(rock_obs_p.ore_quals, ore_obs)
        n_bores = length(rock_obs_p)
        stopped_p = n_bores >= m.max_bores
        decided_p = false
        obs = MEObservation(ore_obs, stopped_p, false)
    else
        error("Invalid Action! Action: $(a.type), Stopped: $stopped, Decided: $decided")
    end
    sp = MEState(s.ore_map, s.mainbody_params, s.mainbody_map, rock_obs_p, stopped_p, decided_p)
    return (sp=sp, o=obs, r=r)
end


function POMDPs.actions(m::MineralExplorationPOMDP)
    idxs = CartesianIndices(m.grid_dim[1:2])
    bore_actions = reshape(collect(idxs), prod(m.grid_dim[1:2]))
    actions = MEAction[MEAction(type=:stop), MEAction(type=:mine),
                        MEAction(type=:abandon)]
    grid_step = m.grid_spacing + 1
    for coord in bore_actions[1:grid_step:end]
        push!(actions, MEAction(coords=coord))
    end
    return actions
end

function POMDPs.actions(m::MineralExplorationPOMDP, s::MEState)
    if s.decided
        return MEAction[]
    elseif s.stopped
        return MEAction[MEAction(type=:mine), MEAction(type=:abandon)]
    else
        action_set = Set(POMDPs.actions(m))
        n_initial = length(m.initial_data)
        n_obs = length(s.rock_obs.ore_quals) - n_initial
        if m.max_movement != 0 && n_obs > 0
            d = m.max_movement
            drill_s = s.rock_obs.coordinates[:,end]
            x = drill_s[1]
            y = drill_s[2]
            reachable_coords = CartesianIndices((x-d:x+d,y-d:y+d))
            reachable_acts = MEAction[]
            for coord in reachable_coords
                dx = abs(x - coord[1])
                dy = abs(y - coord[2])
                d2 = sqrt(dx^2 + dy^2)
                if d2 <= d
                    push!(reachable_acts, MEAction(coords=coord))
                end
            end
            push!(reachable_acts, MEAction(type=:stop))
            reachable_acts = Set(reachable_acts)
            # reachable_acts = Set([MEAction(coords=coord) for coord in collect(reachable_coords)])
            action_set = intersect(reachable_acts, action_set)
        end
        for i=1:n_obs
            coord = s.rock_obs.coordinates[:, i + n_initial]
            x = Int64(coord[1])
            y = Int64(coord[2])
            keepout = collect(CartesianIndices((x-m.delta:x+m.delta,y-m.delta:y+m.delta)))
            keepout_acts = Set([MEAction(coords=coord) for coord in keepout])
            setdiff!(action_set, keepout_acts)
        end
        # delete!(action_set, MEAction(type=:mine))
        # delete!(action_set, MEAction(type=:abandon))
        return collect(action_set)
    end
    return MEAction[]
end

function POMDPModelTools.obs_weight(m::MineralExplorationPOMDP, s::MEState,
                    a::MEAction, sp::MEState, o::MEObservation)
    w = 0.0
    if a.type != :drill
        w = o.ore_quality == nothing ? 1.0 : 0.0
    else
        o_mainbody = s.mainbody_map[a.coords[1], a.coords[2], 1]
        o_gp = (o.ore_quality - o_mainbody)
        mu = m.gp_mean
        sigma = sqrt(m.variogram[1])
        point_dist = Normal(mu, sigma)
        w = pdf(point_dist, o_gp)
    end
    return w
end
