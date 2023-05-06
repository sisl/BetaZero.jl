using Revise
using BetaZero
using Plots; default(fontfamily="Computer Modern", framestyle=:box)
using ParticleFilters
using POMCPOW
using POMDPs
using POMDPTools
using LinearAlgebra
using ParticleBeliefs
using StatsBase
using MinEx

pomdp = MinExPOMDP()
up = ParticleHistoryBeliefUpdater(BootstrapFilter(pomdp, pomdp.N))

function simple_minex_accuracy_func(pomdp::POMDP, b0, s0, states, actions, returns)
    massive = MinEx.calc_massive(pomdp, s0)
    truth = (massive > pomdp.extraction_cost) ? :mine : :abandon
    is_correct = (actions[end] == truth)
    return is_correct
end

simple_minex_belief_reward(pomdp::POMDP, b, a, bp) = mean(reward(pomdp, s, a) for s in MinEx.particles(b.particles))

BetaZero.optimal_return(pomdp::MinExPOMDP, s) = max(0, extraction_reward(pomdp, s))

function compute_optimal_return_minex(pomdp::MinExPOMDP; kwargs...)
    ds0 = initialstate(pomdp)
    if ds0 isa Vector
        # discrete (cached) particle set (Note, avoid using this)
        @warn "Computing using discrete, generated particle set"
        ore_matrix = load_states(joinpath(@__DIR__, "..", "models", "MinEx", "src", "generated_states.h5"))
        ds0 = [MinExState(ore_matrix[:,:,i]) for i in axes(ore_matrix,3)]
    end
    return compute_optimal_return_minex(pomdp, ds0; kwargs...)
end

function compute_optimal_return_minex(pomdp, ds0::MinEx.MinExStateDistribution; n=10_000, kwargs...)
    particles = rand(ds0, n)
    return compute_optimal_return_minex(pomdp, particles; kwargs...)
end

function compute_optimal_return_minex(pomdp, ds0; include_drills=false)
    if include_drills
        # compute optimal returns if fully drilled all locations, then made the oracle mine/abandon decision
        discounted_full_drill_cost = sum(discount(pomdp)^(t-1)*-pomdp.drill_cost for t in 1:length(actions(pomdp))-2)
    else
        # make oracle decision without drilling
        discounted_full_drill_cost = 0
    end
    optimal_r = [discounted_full_drill_cost + BetaZero.optimal_return(pomdp, s) for s in ds0]
    return mean_and_stderr(optimal_r)
end


POMDPs.convert_s(::Type{A}, b::ParticleHistoryBelief{MinExState}, m::BetaZero.BeliefMDP) where A<:AbstractArray = Flux.unsqueeze(Float32.(BetaZero.input_representation(b)); dims=4)
# POMDPs.convert_s(::Type{ParticleHistoryBelief{MinExState}}, b::A, m::BetaZero.BeliefMDP) where A<:AbstractArray = ParticleHistoryBelief(particles=ParticleCollection(rand(LDNormalStateDist(b[1], b[2]), 500)))


zeroifnan(x) = isnan(x) ? 0 : x
data_skewness(D) = [zeroifnan(skewness(D[x,y,1:end-1])) for x in axes(D,1), y in axes(D,2)]
data_kurtosis(D) = [zeroifnan(kurtosis(D[x,y,1:end-1])) for x in axes(D,1), y in axes(D,2)]


function BetaZero.input_representation(b::ParticleHistoryBelief{MinExState};
        use_higher_orders::Bool=false, include_obs::Bool=false)

    states::Vector{MinExState} = particles(b)
    grid_dims::Tuple = size(states[1].ore)
    n_particles::Int = length(states)
    # n_channels::Int = 2 + (use_higher_orders ? 2 : 0) + include_obs
    stacked_states = Array{Float32}(undef, grid_dims..., n_particles)
    for i in 1:n_particles
        stacked_states[:,:,i] = states[i].ore
    end
    μ, σ = mean_and_std(stacked_states, 3)
    return cat(μ, σ; dims=3)
end


## Plotting

function plot_belief(b::ParticleHistoryBelief{MinExState}, s=nothing)
    b̃ = BetaZero.input_representation(b)
    μ = b̃[:,:,1]
    σ = b̃[:,:,2]
    plt_mean = heatmap(μ, ratio=1, c=:viridis, title="\$\\mu(b)\$", clims=(0,1))
    current_ylims = ylims()
    xlims!(current_ylims...)
    drill_style = (label=false, c=:black, mc=:darkred, msc=:white, marker=:square, ms=4)
    if !isnothing(s) && !isempty(s.drill_locations)
        xloc = map(last, s.drill_locations) # Note y-first, x-last
        yloc = map(first, s.drill_locations)
        n = length(xloc)
        cmap = cgrad([:white, :black])
        if n > 1
            for i in 1:n-1
                x = xloc[i:i+1]
                y = yloc[i:i+1]
                c = n == 2 ? get(cmap, 0) : get(cmap, (i-1)/(n-2))
                plot!(x, y, arrow=:closed, lw=2, color=c, label=false)
            end
        end
        scatter!(xloc, yloc; drill_style...)
    end
    ylims!(current_ylims...)

    plt_std =  heatmap(σ, ratio=1, c=:viridis, title="\$\\sigma(b)\$", clims=(0.0, 0.2))
    current_ylims = ylims()
    xlims!(current_ylims...)
    if !isnothing(s) && !isempty(s.drill_locations)
        xloc = map(last, s.drill_locations) # Note y-first, x-last
        yloc = map(first, s.drill_locations)
        scatter!(xloc, yloc; drill_style...)
    end
    ylims!(current_ylims...)

    return plot(plt_mean, plt_std, layout=2, margin=4Plots.mm, size=(1000, 400))
end


function plot_state(s::MinExState)
    heatmap(s.ore, ratio=1, c=:viridis, title="state", clims=(0,1), margin=4Plots.mm, size=(500, 400))
    return xlims!(ylims()...)
end


function plot_trajectory(beliefs::Vector, states::Union{Vector,Nothing}=nothing; filename::Function=i->"belief$i.png", betterfig::Bool=false)
    for i in eachindex(beliefs)
        @info "Plotting belief $i/$(length(beliefs))"
        if isnothing(states)
            plot_belief(beliefs[i])
        else
            plot_belief(beliefs[i], states[i])
        end
        if betterfig
            bettersavefig(filename(i))
        else
            savefig(filename(i))
        end
    end
end


function plot_volume(volume, true_volume=nothing; bins=[-200:10:200;])
	μ, σ = mean_and_std(volume)
    h = fit(Histogram, volume, bins)
    h = normalize(h, mode=:probability)

	rd = x->round(x, digits=2)
    plot(h, title="belief volumes (μ=$(rd(μ)), σ=$(rd(σ)))", label="economic volume", c=:cadetblue)
    h_height = maximum(h.weights)
    ylims!(0, h_height*1.05)

	if !isnothing(true_volume)
		vline!([true_volume], c=:black, ls=:dash, lw=2, label="true volume")
	end
	vline!([μ], c=:crimson, lw=2, label="mean volume")
	vline!([μ - σ], c=:crimson, lw=2, alpha=0.5, ls=:dot, label="standard deviation")
	vline!([μ + σ], c=:crimson, lw=2, alpha=0.5, ls=:dot, label=false)
	plot!(size=(600,350), margin=10Plots.mm, top_margin=5Plots.mm, xlabel="economic volume", ylabel="probability")
end


function plot_volumes(volumes::Vector, true_volume=nothing; filename::Function=i->"volume$i.png", betterfig::Bool=false)
    for i in eachindex(volumes)
        @info "Plotting volume $i/$(length(volumes))"
        plot_volume(volumes[i], true_volume)
        if betterfig
            bettersavefig(filename(i))
        else
            savefig(filename(i))
        end
    end
end
