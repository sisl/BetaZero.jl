Sys.islinux() && include("launch_remote.jl")
using Revise
using Distributed

@everywhere begin
    using Parameters
    using ParticleFilters
    using POMDPModels
    using POMDPs
    using POMDPTools
    using Plots; default(fontfamily="Computer Modern", framestyle=:box)
    using Random
    using Statistics
    using StatsBase
    using BetaZero
    using Flux
end

@everywhere begin
    # import Base: +
    # Base.:(+)(s1::LightDark1DState, s2::LightDark1DState) = LightDark1DState(s1.status + s2.status, s1.y + s2.y)
    # Base.:(/)(s::LightDark1DState, i::Int) = LightDark1DState(s.status, s.y/i)

    POMDPs.actions(::LightDark1D) = [-1, 0, 1] # -10:10 # [-10, -5, -1, 0, 1, 5, 10]

    function POMDPs.transition(p::LightDark1D, s::LightDark1DState, a::Int)
        if a == 0
            return Deterministic(LightDark1DState(-1, s.y+a*p.step_size))
        else
            max_y = 100
            return Deterministic(LightDark1DState(s.status, clamp(s.y+a*p.step_size, -max_y, max_y)))
        end
    end

    function POMDPs.reward(p::LightDark1D, s::LightDark1DState, a::Int)
        if s.status < 0
            return 0.0
        elseif a == 0
            if abs(s.y) < 1
                return p.correct_r
            else
                return p.incorrect_r
            end
        else
            return -p.movement_cost
        end
    end

    @with_kw mutable struct HeuristicLightDarkPolicy <: POMDPs.Policy
        pomdp
        thresh = 0.1
    end

    function POMDPs.action(policy::HeuristicLightDarkPolicy, b::ParticleCollection)
        ỹ = mean(s.y for s in ParticleFilters.particles(b))
        if abs(ỹ) ≤ policy.thresh
            return 0
        else
            A = filter(a->a != 0, actions(policy.pomdp))
            return rand(A)
        end
    end

    #========== BetaZero ==========#
    # Interface:
    # 1) BetaZero.input_representation(b) -> Vector or Matrix
    #============= || =============#

    function BetaZero.input_representation(b::ParticleCollection; use_higher_orders::Bool=true)
        Y = [s.y for s in ParticleFilters.particles(b)]
        μ = mean(Y)
        σ = std(Y)
        if use_higher_orders
            zeroifnan(x) = isnan(x) ? 0 : x
            s = zeroifnan(skewness(Y))
            k = zeroifnan(kurtosis(Y))
            return Float32[μ, σ, s, k]
        else
            return Float32[μ, σ]
        end
    end


    # Simpler MLP
    function BetaZero.initialize_network(nn_params::BetaZeroNetworkParameters) # MLP
        # nσ = 3
        # @info "Using simplified MLP (ReLUs with $(nσ)tanh) for neural network..."
        @info "Using simplified MLP (ReLUs) for neural network..."
        input_size = nn_params.input_size
        layer_dims = [256, 256, 256, 256]
        out_dim = 1

        layers = Dense[Dense(prod(input_size), layer_dims[1], relu)]
        for i in 1:length(layer_dims)-1
            push!(layers, Dense(layer_dims[i], layer_dims[i+1], relu))
        end
        push!(layers, Dense(layer_dims[end], out_dim))
        # push!(layers, Dense(out_dim, out_dim, y->nσ*tanh(y))) # σ on tanh output.

        # Note: A normalization layer will be added during training (with the old layer removed before the next training phase).
        return Chain(layers...)
    end

    lightdark_accuracy_func(pomdp, b0, s0, final_action, returns) = returns[end] == pomdp.correct_r
    lightdark_belief_reward(pomdp, b, a, bp) = mean(reward(pomdp, s, a) for s in ParticleFilters.particles(b))
end

light_loc = 10
sigma = y->abs(y - light_loc) + 1e-4
pomdp = LightDark1D(0.9, 100.0, -100.0, 1.0, 0.0, sigma)
up = BootstrapFilter(pomdp, 500)

if false # !Sys.islinux() && false
    seed = rand(1:100_000)
    @show seed
    Random.seed!(seed) # Determinism (9, 11870, 24269)

    # rand_policy = RandomPolicy(pomdp)
    heuristic_policy = HeuristicLightDarkPolicy(; pomdp)
    policy2use = rand_policy

    S = []
    A = []
    O = []
    B = []
    R = []
    max_steps = 20
    for (s,a,o,b,r,sp,bp) in stepthrough(pomdp, policy2use, up, "s,a,o,b,r,sp,bp"; max_steps)
        ỹ = mean(s.y for s in ParticleFilters.particles(b))
        push!(S, s)
        push!(A, a)
        push!(O, o)
        push!(B, b)
        push!(R, r)
        @info s.y, a, r, ỹ
    end

    G = BetaZero.compute_returns(R; γ=POMDPs.discount(pomdp))
    # display([G [Float64(BetaZero.value_lookup(b, policy.network)) for b in B]])

    function plot_beliefs(B; hold=false)
        !hold && plot()
        for i in eachindex(B)
            n = length(ParticleFilters.particles(B[i]))
            P = ParticleFilters.particles(B[i])
            X = i*ones(n)
            Y = [p.y for p in P]
            scatter!(X, Y, c=:black, msc=:white, alpha=0.25, ms=2, label=i==1 ? "belief" : false)
        end
        return plot!()
    end

    # using ColorSchemes
    Y = map(s->s.y, S)
    Ỹ = map(b->mean(p.y for p in ParticleFilters.particles(b)), B)
    ymax = max(20, max(maximum(Y), abs(minimum(Y))))*1.5
    xmax = max(length(S), max_steps)
    plt_lightdark = plot(xlims=(1, xmax), ylims=(-ymax, ymax), size=(900,200), margin=5Plots.mm, legend=:outertopleft, xlabel="time", ylabel="state")
    heatmap!(1:xmax, range(-ymax, ymax, length=100), (x,y)->sqrt(std(observation(pomdp, LightDark1DState(0, y)))), c=:grayC)
    hline!([0], c=:green, style=:dash, label="goal")
    # plot!(eachindex(S), O, mark=true, ms=2, c=:gray, mc=:black, msc=:white, label="observation")
    plot_beliefs(B; hold=true)
    plot!(eachindex(S), Y, c=:red, lw=2, label="trajectory", alpha=0.5)
    plot!(eachindex(S), Ỹ, c=:blue, lw=1, ls=:dash, label="believed traj.", alpha=0.5)
    scatter!(eachindex(S), O, ms=2, c=:cyan, msc=:black, label="observation")
    display(plt_lightdark)
end



solver = BetaZeroSolver(pomdp=pomdp,
                        updater=up,
                        belief_reward=lightdark_belief_reward,
                        n_iterations=2,
                        n_data_gen=1000,
                        n_evaluate=0, # NOTE.
                        n_holdout=0, # NOTE.
                        use_random_policy_data_gen=false, # NOTE: Hack to use `data_gen_policy`
                        use_onestep_lookahead_holdout=false,
                        # data_gen_policy=HeuristicLightDarkPolicy(; pomdp),
                        collect_metrics=true,
                        verbose=true,
                        include_info=false,
                        accuracy_func=lightdark_accuracy_func)

solver.n_buffer = solver.n_iterations # NOTE: solver.n_iterations

solver.mcts_solver.n_iterations = 100
solver.mcts_solver.exploration_constant = 1.0 # NOTE: 2.0
solver.mcts_solver.k_state = 2.0
# solver.mcts_solver.k_action = 1.0

solver.onestep_solver.n_actions = 20
solver.onestep_solver.n_obs = 2

solver.network_params.training_epochs = 1000
solver.network_params.n_samples = 5_000 # NOTE: 1000
solver.network_params.verbose_plot_frequency = solver.network_params.training_epochs + 1
solver.network_params.verbose_update_frequency = 100
solver.network_params.learning_rate = 0.007 # NOTE: 0.005 (better validation: 0.001)
solver.network_params.batchsize = 256
solver.network_params.λ_regularization = 1e-4 # NOTE: 0.00001 (better validation: 0.0001, even better validation: 0.001)
solver.network_params.normalize_target = true
# solver.network_params.loss_func = Flux.Losses.mse # NOTE: mean-squared error

policy = solve(solver, pomdp)
# BetaZero.save_policy(policy, "policy_lightdark.bson")


# results = BetaZero.tune_network_parameters(pomdp, solver; learning_rates=range(1e-1, 1e-7, length=10), λs=range(1e-1, 1e-7, length=10))

function kldist(model, data; ϵ=1e-10)
    mfit = fit(Histogram, vec(model))
    dfit = fit(Histogram, vec(data), mfit.edges[1])
    return kldivergence(mfit.weights .+ ϵ, dfit.weights .+ ϵ)
end

function hyperopt_tune(pomdp::POMDP, solver::BetaZeroSolver;
                       learning_rate_coeffs=1:9,
                       learning_rate_exponents=-6:-2,
                       λs=[0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                       loss_funcs=[Flux.Losses.mae, Flux.Losses.mse],
                       normalize_targets=[true, false],
                       resources=10)
    # _use_random_policy_data_gen = solver.use_random_policy_data_gen # save original setting
    # solver.use_random_policy_data_gen = true
    # @info "Tuning using a random policy for data generation."
    results = Dict()
    ho = Hyperoptimizer(resources, lrc=learning_rate_coeffs, lre=learning_rate_exponents, λ=λs, loss_func=loss_funcs, normalize_target=normalize_targets)
    for (i, lrc, lre, λ, loss_func, normalize_target) in ho
        lr = lrc * 10.0^lre
        @info "Tuning iteration: $i/$resources ($(round(i/resources*100, digits=3)))"
        solver.network_params.learning_rate = lr
        solver.network_params.λ_regularization = λ
        solver.network_params.loss_func = loss_func
        solver.network_params.normalize_target = normalize_target
        loss_str = string(loss_func)

        @info "Tuning with: lr=$lr, λ=$λ, loss=$loss_str, normalize_target=$normalize_target"
        empty!(solver.data_buffer)
        f_prev = BetaZero.initialize_network(solver)
        BetaZero.generate_data!(pomdp, solver, f_prev; use_random_policy=solver.use_random_policy_data_gen, inner_iter=solver.n_data_gen, outer_iter=1)
        f_curr = BetaZero.train_network(deepcopy(f_prev), solver; verbose=solver.verbose, results=results)

        key = (lr, λ, loss_str, normalize_target)
        results[key]["network"] = f_curr
        results[key]["kl"] = kldist(results[key]["value_model"], results[key]["value_data"])
    end
    # solver.use_random_policy_data_gen = _use_random_policy_data_gen # reset to original setting
    return results
end

# results = hyperopt_tune(pomdp, solver; learning_rate_coeffs=[7], learning_rate_exponents=[-4], normalize_targets=[true], loss_funcs=[Flux.Losses.mae], resources=10)


# TRY: lr=0.005, λ=0.1, loss=mse
# TRY: lr=0.0001, λ=0.001, loss=mae
# TRY: lr=0.0001, λ=0.0001, loss=mse


# # other_solver = OneStepLookaheadSolver(n_actions=100,
#                                 # n_obs=10)
# other_solver = solver.mcts_solver
# bmdp = BeliefMDP(pomdp, up, lightdark_belief_reward)
# policy = solve(other_solver, bmdp)


# # policy = HeuristicLightDarkPolicy(; pomdp)
# b0 = initialize_belief(up, [rand(initialstate(pomdp)) for _ in 1:up.n_init])
# s0 = rand(initialstate(pomdp))
# @time data, metrics = BetaZero.run_simulation(pomdp, policy, up, b0, s0; accuracy_func=lightdark_accuracy_func, collect_metrics=true, include_info=true, max_steps=1000); metrics.accuracy


if false
    init_network = BetaZero.initialize_network(solver)
    @time _B = [Float32.(BetaZero.input_representation(initialize_belief(up, [rand(initialstate(pomdp)) for _ in 1:up.n_init]))) for _ in 1:100]
    @time _B = cat(_B...; dims=2)

    @time returns0_init_network = init_network(_B)
    @time returns0 = policy.network(_B)

    # network = BetaZero.initialize_network(solver)
    # @time returns0 = network(_B)

    network = BetaZero.train_network(deepcopy(network), solver; verbose=true)

    histogram(returns0', label="learned model", alpha=0.5)
    histogram!(returns0', label="uninitialized model", alpha=0.5)
end
