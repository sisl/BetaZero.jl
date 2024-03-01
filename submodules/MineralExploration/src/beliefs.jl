
struct MEBelief{G}
    particles::Vector{MEState} # Vector of vars & lode maps
    rock_obs::RockObservations
    acts::Vector{MEAction}
    obs::Vector{MEObservation}
    stopped::Bool
    decided::Bool
    geostats::G #GSLIB or GeoStats
end

struct MEBeliefUpdater{G} <: POMDPs.Updater
    m::MineralExplorationPOMDP
    geostats::G
    n::Int64
    noise::Float64
    rng::AbstractRNG
end

function MEBeliefUpdater(m::MineralExplorationPOMDP, n::Int64, noise::Float64=1.0)
    geostats = m.geodist_type(m)
    return MEBeliefUpdater(m, geostats, n, noise, Random.GLOBAL_RNG)
end

function POMDPs.initialize_belief(up::MEBeliefUpdater, d::MEInitStateDist)
    particles = rand(d, up.n)
    init_rocks = up.m.initial_data
    rock_obs = RockObservations(init_rocks.ore_quals, init_rocks.coordinates)
    acts = MEAction[]
    obs = MEObservation[]
    return MEBelief(particles, rock_obs, acts, obs, false, false, up.geostats)
end

function calc_K(geostats::GeoDist, rock_obs::RockObservations)
    if isa(geostats, GeoStatsDistribution)
        pdomain = geostats.domain
        γ = geostats.variogram
    else
        pdomain = CartesianGrid{Int64}(geostats.grid_dims[1], geostats.grid_dims[2])
        γ = SphericalVariogram(sill=geostats.sill, range=geostats.variogram[6], nugget=geostats.nugget)
    end
    table = DataFrame(ore=rock_obs.ore_quals .- geostats.mean)
    domain = PointSet(rock_obs.coordinates)
    pdata = georef(table, domain)
    vmapping = map(pdata, pdomain, (:ore,), GeoStats.NearestMapping())[:ore]
    # dlocs = Int[]
    # for (loc, dloc) in vmapping
    #     push!(dlocs, loc)
    # end
    dlocs = Int64[m[1] for m in vmapping]
    𝒟d = [centroid(pdomain, i) for i in dlocs]
    K = GeoStats.sill(γ) .- GeoStats.pairwise(γ, 𝒟d)
    return K
end

function reweight(up::MEBeliefUpdater, geostats::GeoDist, particles::Vector, rock_obs::RockObservations)
    ws = Float64[]
    bore_coords = rock_obs.coordinates
    n = size(bore_coords)[2]
    ore_obs = [o for o in rock_obs.ore_quals]
    K = calc_K(geostats, rock_obs)
    mu = zeros(Float64, n) .+ up.m.gp_mean
    gp_dist = MvNormal(mu, K)
    for s in particles
        mb_map = s.mainbody_map
        o_n = zeros(Float64, n)
        for i = 1:n
            o_mainbody = mb_map[bore_coords[1, i], bore_coords[2, i]]
            o_n[i] = ore_obs[i] - o_mainbody
        end
        w = pdf(gp_dist, o_n)
        push!(ws, w)
    end
    ws .+= 1e-6
    ws ./= sum(ws)
    return ws
end

function resample(up::MEBeliefUpdater, particles::Vector, wp::Vector{Float64},
                geostats::GeoDist, rock_obs::RockObservations, a::MEAction, o::MEObservation)
    sampled_particles = sample(up.rng, particles, StatsBase.Weights(wp), up.n, replace=true)
    mainbody_params = []
    mainbody_maps = Array{Float64, 3}[]
    particles = MEState[]
    x = nothing
    ore_quals = deepcopy(rock_obs.ore_quals)
    for s in sampled_particles
        mainbody_param = s.mainbody_params
        mainbody_map = s.mainbody_map
        ore_map = s.ore_map
        if mainbody_param ∈ mainbody_params
            mainbody_map, mainbody_param = perturb_sample(up.m.mainbody_gen, mainbody_param, up.noise)
            max_lode = maximum(mainbody_map)
            mainbody_map ./= max_lode
            mainbody_map .*= up.m.mainbody_weight
            mainbody_map = reshape(mainbody_map, up.m.grid_dim)
            # clamp!(ore_map, 0.0, 1.0)
        end
        n_ore_quals = Float64[]
        for (i, ore_qual) in enumerate(ore_quals)
            prior_ore = mainbody_map[rock_obs.coordinates[1, i], rock_obs.coordinates[2, i], 1]
            n_ore_qual = (ore_qual - prior_ore)
            push!(n_ore_quals, n_ore_qual)
        end
        geostats.data.ore_quals = n_ore_quals
        # gslib_dist.data.ore_quals = n_ore_quals
        gp_ore_map = Base.rand(up.rng, geostats)
        ore_map = gp_ore_map .+ mainbody_map
        rock_obs_p = RockObservations(rock_obs.ore_quals, rock_obs.coordinates)
        sp = MEState(ore_map, mainbody_param, mainbody_map, rock_obs_p,
                    o.stopped, o.decided)
        push!(mainbody_params, mainbody_param)
        push!(particles, sp)
    end
    return particles
end

function update_particles(up::MEBeliefUpdater, particles::Vector{MEState},
                        geostats::GeoDist, rock_obs::RockObservations, a::MEAction, o::MEObservation)
    wp = reweight(up, geostats, particles, rock_obs)
    pp = resample(up, particles, wp, geostats, rock_obs, a, o)
    return pp
end

function POMDPs.update(up::MEBeliefUpdater, b::MEBelief,
                            a::MEAction, o::MEObservation)
    if a.type != :drill
        bp_particles = MEState[] # MEState[p for p in b.particles]
        for p in b.particles
            s = MEState(p.ore_map, p.mainbody_params, p.mainbody_map, p.rock_obs, o.stopped, o.decided)
            push!(bp_particles, s)
        end
        bp_rock = RockObservations(ore_quals=deepcopy(b.rock_obs.ore_quals),
                                coordinates=deepcopy(b.rock_obs.coordinates))
        # TODO Make this a little more general in future
        if up.m.geodist_type == GeoStatsDistribution
            bp_geostats = GeoStatsDistribution(b.geostats.grid_dims, bp_rock,
                                            b.geostats.domain, b.geostats.mean,
                                            b.geostats.variogram, b.geostats.lu_params)
        elseif up.m.geodist_type == GSLIBDistribution
            bp_geostats = GSLIBDistribution(b.geostats.grid_dims, b.geostats.grid_dims,
                                            bp_rock, b.geostats.mean, b.geostats.sill, b.geostats.nugget,
                                            b.geostats.variogram, b.geostats.target_histogram_file,
                                            b.geostats.columns_for_vr_and_wt, b.geostats.zmin_zmax,
                                            b.geostats.lower_tail_option, b.geostats.upper_tail_option,
                                            b.geostats.transform_data, b.geostats.mn,
                                            b.geostats.sz)
        end
    else
        bp_rock = deepcopy(b.rock_obs)
        bp_rock.coordinates = hcat(bp_rock.coordinates, [a.coords[1], a.coords[2]])
        push!(bp_rock.ore_quals, o.ore_quality)
        if up.m.geodist_type == GeoStatsDistribution
            bp_geostats = GeoStatsDistribution(b.geostats.grid_dims, deepcopy(bp_rock),
                                            b.geostats.domain, b.geostats.mean,
                                            b.geostats.variogram, b.geostats.lu_params)
            update!(bp_geostats, bp_rock)
        elseif up.m.geodist_type == GSLIBDistribution
            bp_geostats = GSLIBDistribution(b.geostats.grid_dims, b.geostats.grid_dims,
                                            bp_rock, b.geostats.mean, b.geostats.sill, b.geostats.nugget,
                                            b.geostats.variogram, b.geostats.target_histogram_file,
                                            b.geostats.columns_for_vr_and_wt, b.geostats.zmin_zmax,
                                            b.geostats.lower_tail_option, b.geostats.upper_tail_option,
                                            b.geostats.transform_data, b.geostats.mn,
                                            b.geostats.sz)
        end
        bp_particles = update_particles(up, b.particles, bp_geostats, bp_rock, a, o)
    end

    bp_acts = MEAction[]
    for act in b.acts
        push!(bp_acts, act)
    end
    push!(bp_acts, a)

    bp_obs = MEObservation[]
    for obs in b.obs
        push!(bp_obs, obs)
    end
    push!(bp_obs, o)

    bp_stopped = o.stopped
    bp_decided = o.decided

    return MEBelief(bp_particles, bp_rock, bp_acts, bp_obs, bp_stopped,
                    bp_decided, bp_geostats)
end

function Base.rand(rng::AbstractRNG, b::MEBelief)
    return rand(rng, b.particles)
end

Base.rand(b::MEBelief) = rand(Random.GLOBAL_RNG, b)

function summarize(b::MEBelief)
    (x, y, z) = size(b.particles[1].ore_map)
    μ = zeros(Float64, x, y, z)
    w = 1.0/length(b.particles)
    for p in b.particles
        ore_map = p.ore_map
        μ .+= ore_map .* w
    end
    σ² = zeros(Float64, x, y, z)
    for p in b.particles
        ore_map = p.ore_map
        σ² .+= w*(ore_map - μ).^2
    end
    return (μ, σ²)
end

function POMDPs.actions(m::MineralExplorationPOMDP, b::MEBelief)
    if b.stopped
        return MEAction[MEAction(type=:mine), MEAction(type=:abandon)]
    else
        action_set = Set(POMDPs.actions(m))
        n_initial = length(m.initial_data)
        if !isempty(b.rock_obs.ore_quals)
            n_obs = length(b.rock_obs.ore_quals) - n_initial
            if m.max_movement != 0 && n_obs > 0
                d = m.max_movement
                drill_s = b.rock_obs.coordinates[:,end]
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
                coord = b.rock_obs.coordinates[:, i + n_initial]
                x = Int64(coord[1])
                y = Int64(coord[2])
                keepout = collect(CartesianIndices((x-m.delta:x+m.delta,y-m.delta:y+m.delta)))
                keepout_acts = Set([MEAction(coords=coord) for coord in keepout])
                setdiff!(action_set, keepout_acts)
            end
            if n_obs < m.min_bores
                delete!(action_set, MEAction(type=:stop))
            end
        elseif m.min_bores > 0
            delete!(action_set, MEAction(type=:stop))
        end
        delete!(action_set, MEAction(type=:mine))
        delete!(action_set, MEAction(type=:abandon))
        return collect(action_set)
    end
    return MEAction[]
end

function POMDPs.actions(m::MineralExplorationPOMDP, b::POMCPOW.StateBelief)
    o = b.sr_belief.o
    s = rand(b.sr_belief.dist)[1]
    if o.stopped
        return MEAction[MEAction(type=:mine), MEAction(type=:abandon)]
    else
        action_set = Set(POMDPs.actions(m))
        n_initial = length(m.initial_data)
        if !isempty(s.rock_obs.ore_quals)
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
            if n_obs < m.min_bores
                delete!(action_set, MEAction(type=:stop))
            end
        elseif m.min_bores > 0
            delete!(action_set, MEAction(type=:stop))
        end
        # delete!(action_set, MEAction(type=:mine))
        # delete!(action_set, MEAction(type=:abandon))
        return collect(action_set)
    end
    return MEAction[]
end

function POMDPs.actions(m::MineralExplorationPOMDP, o::MEObservation)
    if o.stopped
        return MEAction[MEAction(type=:mine), MEAction(type=:abandon)]
    else
        action_set = Set(POMDPs.actions(m))
        delete!(action_set, MEAction(type=:mine))
        delete!(action_set, MEAction(type=:abandon))
        return collect(action_set)
    end
    return MEAction[]
end

function mean_var(b::MEBelief)
    vars = [s[1] for s in b.particles]
    mean(vars)
end

function std_var(b::MEBelief)
    vars = [s[1] for s in b.particles]
    std(vars)
end

function Plots.plot(b::MEBelief, t=nothing)
    mean, var = summarize(b)
    if t == nothing
        mean_title = "Belief Mean"
        std_title = "Belief StdDev"
    else
        mean_title = "Belief Mean t=$t"
        std_title = "Belief StdDev t=$t"
    end
    fig1 = heatmap(mean[:,:,1], title=mean_title, fill=true, clims=(0.0, 1.0), legend=:none)
    fig2 = heatmap(sqrt.(var[:,:,1]), title=std_title, fill=true, legend=:none, clims=(0.0, 0.2))
    if !isempty(b.rock_obs.ore_quals)
        x = b.rock_obs.coordinates[2, :]
        y = b.rock_obs.coordinates[1, :]
        plot!(fig1, x, y, seriestype = :scatter)
        plot!(fig2, x, y, seriestype = :scatter)
        n = length(b.rock_obs)
        if n > 1
            for i = 1:n-1
                x = b.rock_obs.coordinates[2, i:i+1]
                y = b.rock_obs.coordinates[1, i:i+1]
                plot!(fig1, x, y, arrow=:closed, color=:black)
            end
        end
    end
    fig = plot(fig1, fig2, layout=(1,2))
    return fig
end
