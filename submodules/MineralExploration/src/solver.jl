@with_kw struct NextActionSampler
    ucb::Float64 = 1.0
end

function sample_ucb_drill(mean, var, idxs)
    scores = belief_scores(mean, var)
    weights = Float64[]
    for idx in idxs
        push!(weights, scores[idx])
    end
    coords = sample(idxs, StatsBase.Weights(weights))
    return MEAction(coords=coords)
end

function belief_scores(m, v)
    norm_mean = m[:,:,1]./(maximum(m[:,:,1]) - minimum(m[:,:,1]))
    norm_mean .-= minimum(norm_mean)
    s = v[:,:,1]
    norm_std = s./(maximum(s) - minimum(s)) # actualy using variance
    norm_std .-= minimum(norm_std)
    scores = (norm_mean .* norm_std).^2
    # scores = norm_mean .+ norm_std
    # scores .+= 1.0/length(scores)
    scores ./= sum(scores)
    return scores
end

function POMCPOW.next_action(o::NextActionSampler, pomdp::MineralExplorationPOMDP,
                            b::MEBelief, h)
    tried_idxs = h.tree.tried[h.node]
    action_set = POMDPs.actions(pomdp, b)
    if b.stopped
        if length(tried_idxs) == 0
            return MEAction(type=:abandon)
        else
            return MEAction(type=:mine)
        end
    else
        volumes = Float64[]
        for s in b.particles
            s_massive = s.ore_map .>= pomdp.massive_threshold
            v = sum(s_massive)
            push!(volumes, v)
        end
        # volumes = Float64[sum(p[2]) for p in b.particles]
        mean_volume = Statistics.mean(volumes)
        volume_std = Statistics.std(volumes)
        lcb = mean_volume - volume_std*pomdp.extraction_lcb
        ucb = mean_volume + volume_std*pomdp.extraction_ucb
        stop_bound = lcb >= pomdp.extraction_cost || ucb <= pomdp.extraction_cost
        if MEAction(type=:stop) ∈ action_set && length(tried_idxs) <= 0 && stop_bound
            return MEAction(type=:stop)
        else
            mean, var = summarize(b)
            coords = [a.coords for a in action_set if a.type == :drill]
            return sample_ucb_drill(mean, var, coords)
        end
    end
end

function POMCPOW.next_action(obj::NextActionSampler, pomdp::MineralExplorationPOMDP,
                            b::POMCPOW.StateBelief, h)
    o = b.sr_belief.o
    # s = rand(b.sr_belief.dist)[1]
    tried_idxs = h.tree.tried[h.node]
    action_set = POMDPs.actions(pomdp, b)
    if o.stopped
        if length(tried_idxs) == 0
            return MEAction(type=:abandon)
        else
            return MEAction(type=:mine)
        end
    else
        if MEAction(type=:stop) ∈ action_set && length(tried_idxs) <= 0
            return MEAction(type=:stop)
        else
            ore_maps = Array{Float64, 3}[]
            weights = Float64[]
            for (idx, item) in enumerate(b.sr_belief.dist.items)
                weight = b.sr_belief.dist.cdf[idx]
                state = item[1]
                push!(ore_maps, state.mainbody_map)
                push!(weights, weight)
            end
            weights ./= sum(weights)
            mean = sum(weights.*ore_maps)
            var = sum([weights[i]*(ore_map - mean).^2 for (i, ore_map) in enumerate(ore_maps)])
            coords = [a.coords for a in action_set if a.type == :drill]
            return sample_ucb_drill(mean, var, coords)
        end
    end
end

struct ExpertPolicy <: Policy
    m::MineralExplorationPOMDP
end

POMCPOW.updater(p::ExpertPolicy) = MEBeliefUpdater(p.m, 1)

function POMCPOW.BasicPOMCP.extract_belief(p::MEBeliefUpdater, node::POMCPOW.BeliefNode)
    srb = node.tree.sr_beliefs[node.node]
    cv = srb.dist
    particles = MEState[]
    weights = Float64[]
    state = nothing
    coords = nothing
    stopped = false
    for (idx, item) in enumerate(cv.items)
        weight = cv.cdf[idx]
        state = item[1]
        coords = state.bore_coords
        stopped = state.stopped
        push!(particles, state)
        push!(weights, weight)
    end
    acts = MEAction[]
    obs = MEObservation[]
    for i = 1:size(state.bore_coords)[2]
        a = MEAction(coords=CartesianIndex((state.bore_coords[1, i], state.bore_coords[2, i])))
        ore_qual = state.ore_map[state.bore_coords[1, i], state.bore_coords[2, i], 1]
        o = MEObservation(ore_qual, state.stopped, state.decided)
        push!(acts, a)
        push!(obs, o)
    end
    return MEBelief(coords, stopped, particles, acts, obs)
end

function POMDPs.action(p::ExpertPolicy, b::MEBelief)
    volumes = Float64[]
    for s in b.particles
        s_massive = s.ore_map .>= p.m.massive_threshold
        v = sum(s_massive)
        push!(volumes, v)
    end
    # volumes = Float64[sum(p[2]) for p in b.particles]
    mean_volume = Statistics.mean(volumes)
    volume_var = Statistics.var(volumes)
    volume_std = sqrt(volume_var)
    lcb = mean_volume - volume_std*p.m.extraction_lcb
    ucb = mean_volume + volume_std*p.m.extraction_ucb
    stop_bound = lcb >= p.m.extraction_cost || ucb <= p.m.extraction_cost
    if b.stopped
        if lcb >= p.m.extraction_cost
            return MEAction(type=:mine)
        else
            return MEAction(type=:abandon)
        end
    elseif stop_bound
        return MEAction(type=:stop)
    else
        ore_maps = Array{Float64, 3}[s.ore_map for s  in b.particles]
        w = 1.0/length(ore_maps)
        mean = sum(ore_maps)./length(ore_maps)
        var = sum([w*(ore_map - mean).^2 for (i, ore_map) in enumerate(ore_maps)])
        action_set = POMDPs.actions(p.m, b)
        coords = [a.coords for a in action_set if a.type == :drill]
        return sample_ucb_drill(mean, var, coords)
    end
end

mutable struct RandomSolver <: POMDPs.Solver
    rng::AbstractRNG
end

RandomSolver(;rng=Random.GLOBAL_RNG) = RandomSolver(rng)
POMDPs.solve(solver::RandomSolver, problem::Union{POMDP,MDP}) = POMCPOW.RandomPolicy(solver.rng, problem, BeliefUpdaters.PreviousObservationUpdater())

function leaf_estimation(pomdp::MineralExplorationPOMDP, s::MEState, h::POMCPOW.BeliefNode, ::Any)
    if s.stopped
        γ = POMDPs.discount(pomdp)
    else
        if isempty(s.rock_obs.ore_quals)
            bores = 0
        else
            bores = length(s.rock_obs.ore_quals)
        end
        t = pomdp.max_bores - bores + 1
        γ = POMDPs.discount(pomdp)^t
    end
    if s.decided
        return 0.0
    else
        r_extract = extraction_reward(pomdp, s)
        if r_extract >= 0.0
            return γ*r_extract*0.9
        else
            return γ*r_extract*0.1
        end
        # return γ*r_extract
    end
end

struct GridPolicy <: Policy
    m::MineralExplorationPOMDP
    n::Int64 # Number of grid points per dimension (n x n)
    grid_size::Int64 # Size of grid area, centered on map center
    grid_coords::Vector{CartesianIndex{2}}
end

function GridPolicy(m::MineralExplorationPOMDP, n::Int, grid_size::Int)
    grid_start_i = (m.grid_dim[1] - grid_size)/2
    grid_start_j = (m.grid_dim[2] - grid_size)/2
    grid_end_i = grid_start_i + grid_size
    grid_end_j = grid_start_j + grid_size
    grid_i = LinRange(grid_start_i, grid_end_i, n)
    grid_j = LinRange(grid_start_j, grid_end_j, n)

    coords = CartesianIndex{2}[]
    for i=1:n
        for j=1:n
            coord = CartesianIndex(Int(floor(grid_i[i])), Int(floor(grid_j[j])))
            push!(coords, coord)
        end
    end
    return GridPolicy(m, n, grid_size, coords)
end

function POMDPs.action(p::GridPolicy, b::MEBelief)
    n_bores = length(b.rock_obs)
    if b.stopped
        volumes = Float64[]
        for s in b.particles
            v = sum(s.ore_map[:, :, 1] .>= p.m.massive_threshold)
            push!(volumes, v)
        end
        mean_volume = Statistics.mean(volumes)
        volume_var = Statistics.var(volumes)
        volume_std = sqrt(volume_var)
        lcb = mean_volume - volume_std*p.m.extraction_lcb
        if lcb >= p.m.extraction_cost
            return MEAction(type=:mine)
        else
            return MEAction(type=:abandon)
        end
    elseif n_bores >= p.n^2
        return MEAction(type=:stop)
    else
        coords = p.grid_coords[n_bores + 1]
        return MEAction(coords=coords)
    end
end
