using Revise
using Distributed

σ_abc_sweep = [0.1, 0.2, 0.3, 0.4]
n_particles_sweep = [1_000, 2_000, 3_000, 10_000]
sweeping_parameters = [(σ_abc, n_particles) for σ_abc in σ_abc_sweep for n_particles in n_particles_sweep]

desired_procs = min(length(sweeping_parameters) + 1, 20)
nprocs() < desired_procs && addprocs(desired_procs - nprocs())
@info "Number of processes: $(nprocs())"

@everywhere begin
    using BetaZero
    using MCTS
    using ParticleFilters
    using Plots; default(fontfamily="Computer Modern", framestyle=:box)
    using POMCPOW
    using POMDPs
    using POMDPTools
    using Random
    using ParticleBeliefs
    using StatsBase
    using MinEx

    include(joinpath(@__DIR__, "..", "..", "..", "scripts", "representation_minex.jl"))
end

Random.seed!(0)
n_runs = 100


@time parallel_results = pmap(i->begin
    σ_abc, n_particles = sweeping_parameters[i]
    pomdp = MinExPOMDP(; σ_abc, n_particles)
    up = ParticleHistoryBeliefUpdater(BootstrapFilter(pomdp, pomdp.n_particles))

    heuristic_policy = MinExHeuristicPolicy(pomdp)
    max_steps = length(actions(pomdp))

    @time ret_acc_results = [begin
        history = simulate(HistoryRecorder(max_steps=max_steps), pomdp, heuristic_policy, up)
        G = discounted_reward(history)
        accuracy = simple_minex_accuracy_func(pomdp, history[end].b, history[end].s, history[end].a, G)
        # @info i, G, accuracy, extraction_reward(pomdp, s0)
        G, accuracy
    end for i in 1:n_runs]
    Gμ, Gσ = mean_and_stderr(first.(ret_acc_results))
    accμ, accσ = mean_and_stderr(last.(ret_acc_results))
    @info "Returns: $Gμ ± $(Gσ/sqrt(n_runs)) \t | \t Accuracy: $accμ ± $(accσ/sqrt(n_runs))"
    Pair((σ_abc, n_particles), ret_acc_results)
end, eachindex(sweeping_parameters))

results = Dict(parallel_results...)

begin
    markers = filter((m->begin
                m in Plots.supported_markers()
            end), Plots._shape_keys)
    plot(palette=cgrad(:rainbow, rev=true, categorical=true))
    for (k,v) in sort(results)
        σ_abc, n_particles = k
        Gμ, Gσ = mean_and_stderr(first.(v))
        accμ, accσ = mean_and_stderr(last.(v))
        color = findfirst(n_particles .== n_particles_sweep)
        i_marker = findfirst(σ_abc .== σ_abc_sweep)
        scatter!([Gμ], [accμ], label="\$($σ_abc, $n_particles)\$", legend=:outerright, marker=markers[i_marker], ms=7, c=color, msc=color)
    end
    plot!(xlabel="mean returns", ylabel="accuracy", title="sweep ABC particle filter (\$\\sigma, n_particles\$)",
          left_margin=10Plots.mm, bottom_margin=10Plots.mm, ylims=(0,1.05))
    bettersavefig("sweep_abc.png")
end
