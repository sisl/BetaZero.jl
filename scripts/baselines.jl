using Distributed
using Revise
@everywhere LAUNCH_PARALLEL = false

if LAUNCH_PARALLEL
    desired_procs = 20
    nprocs() < desired_procs && addprocs(desired_procs)
end

@everywhere begin
    using LightDark
    using RockSample
    using MinEx
    using CollisionAvoidancePOMDPs
    using SpillpointPOMDP
    using LinearAlgebra
end

ENV["LAUNCH_PARALLEL"] = false

@everywhere TestPOMDPType = LightDarkPOMDP # Change this.
@info TestPOMDPType

if TestPOMDPType == LightDarkPOMDP
    include("launch_lightdark.jl")
elseif TestPOMDPType == RockSamplePOMDP
    include("launch_rocksample.jl")
elseif TestPOMDPType == MinExPOMDP
    include("launch_minex.jl")
elseif TestPOMDPType == CollisionAvoidancePOMDP
    include("launch_cas.jl")
elseif TestPOMDPType == SpillpointPOMDP
    include("launch_spillpoint.jl")
end


@everywhere begin
    # TODO: Baselines module or cleaned-up file.
    using Revise
    using POMCPOW
    using ProgressMeter
    using BetaZero
    using ParticleBeliefs
    using Random
    using AdaOPS
    using ARDESPOT
    using StatsBase
    using POMDPs
    using POMDPTools
    using LocalApproximationRandomStrategy # https://github.com/LAMDA-POMDP/LocalApproximationRandomStrategy.jl
    using StaticArrays
    using LightDark
    using RockSample
    using MinEx

    include("init_param.jl")

    function extract_mcts(solver, pomdp)
        mcts_solver = deepcopy(solver.mcts_solver)
        mcts_solver.estimate_value = 0.0
        if hasfield(typeof(mcts_solver), :estimate_failure)
            @info "Estimate failure = 0"
            mcts_solver.estimate_failure = 0.0
            mcts_solver.init_F = 0.0
        end
        mcts_solver.next_action = RandomActionGenerator()
        mcts_solver.estimate_policy = (bmdp, b)->ones(length(actions(pomdp)))
        mcts_solver.reset_callback = (bmdp, s)->false
        mcts_solver.timer = ()->1e-9 * time_ns()
        planner = solve(mcts_solver, solver.bmdp)
        return planner
    end

    function extract_mcts_rand_values(solver, pomdp)
        mcts_solver = deepcopy(solver.mcts_solver)
        mcts_solver.estimate_value = (bmdp,b,d)->rand()
        mcts_solver.next_action = RandomActionGenerator()
        planner = solve(mcts_solver, solver.bmdp)
        return planner
    end

    function solve_pomcpow(pomdp, n_iterations; tree_in_info=false, override=false, use_heuristics=true)
        timing_results = @elapsed begin
            if pomdp isa LightDarkPOMDP
                # From: https://github.com/LAMDA-POMDP/Test/
                if use_heuristics
                    interp = LocalGIFunctionApproximator(RectangleGrid(range(-1, stop=1, length=3), range(-100, stop=100, length=401)))
                    approx_mdp = solve(LocalApproximationValueIterationSolver(
                        interp,
                        verbose=true,
                        max_iterations=1000,
                        is_mdp_generative=true,
                        n_generative_samples=1000),
                        pomdp)
                    value_estimator = FOValue(approx_mdp)
                else
                    @warn "Not using POMCPOW heuristics"
                    value_estimator = AdaOPS.RolloutEstimator(RandomSolver())
                end
                if override
                    @warn "Overriding POMCPOW iterations"
                    n_iterations = 100_000 # From AdaOPS paper.
                end
                pomcpow_solver = POMCPOWSolver(;
                    tree_queries=n_iterations,
                    estimate_value=value_estimator,
                    max_time=1.0,
                    criterion=MaxUCB(10.0),
                    max_depth=20,
                    k_observation=4.0,
                    alpha_observation=0.03,
                )
            elseif pomdp isa RockSamplePOMDP
                if use_heuristics
                    move_east = RSExitSolver()
                    value_estimator = FOValue(move_east)
                else
                    @warn "Not using POMCPOW heuristics"
                    value_estimator = AdaOPS.RolloutEstimator(RandomSolver())
                end
                if override
                    @warn "Overriding POMCPOW iterations"
                    n_iterations = 200_000
                end
                pomcpow_solver = POMCPOWSolver(;
                tree_queries=n_iterations,
                    estimate_value=value_estimator,
                    max_time=1.0,
                    enable_action_pw=false,
                    criterion=MaxUCB(10.0),
                    k_observation=1.0,
                    alpha_observation=1.0,
                )
            elseif pomdp isa MinExPOMDP
                if use_heuristics
                    value_estimator = (pomdp, s, h, steps) -> isterminal(pomdp, s) ? 0 : max(0, MinEx.extraction_reward(pomdp, s))
                else
                    @warn "Not using POMCPOW heuristics"
                    value_estimator = AdaOPS.RolloutEstimator(RandomSolver())
                end
                if override
                    @warn "Overriding POMCPOW iterations"
                    n_iterations = 100_000
                end
                pomcpow_solver = POMCPOWSolver(tree_queries=n_iterations,
                    criterion=POMCPOW.MaxUCB(100.0),
                    k_action=4.0,
                    alpha_action=0.5,
                    k_observation=2.0,
                    alpha_observation=0.25,
                    estimate_value=value_estimator,
                    tree_in_info=tree_in_info,
                )
            end
            pomcpow_planner = solve(pomcpow_solver, pomdp)
        end
        @info timing_results
        return pomcpow_planner
    end

    function adjust_betazero_mcts(mcts_solver; use_bootstrap=false, zq=nothing, zn=nothing, zqn_argmax=true, k_state=nothing, alpha_state=nothing, k_action=nothing, alpha_action=nothing, depth=nothing)
        if !isnothing(zq) && !isnothing(zn)
            if zq == zn == 0
                @warn "We will sample from ZQN instead because zq=zn=0 (i.e., sample uniformly across action space)"
                mcts_solver.final_criterion = MCTS.SampleZQN(; zq, zn)
            else
                if zqn_argmax
                    mcts_solver.final_criterion = MCTS.MaxZQN(; zq, zn)
                else
                    @warn "Sampling instead of taking argmax of ZQN."
                    # We instead want to test sampling the policy during evaluation.
                    τ = mcts_solver.final_criterion.τ
                    mcts_solver.final_criterion = MCTS.SampleZQN(; τ, zq, zn)
                end
            end
        else
            if mcts_solver.final_criterion isa MCTS.SampleZQN
                @info "Changing final_criterion from SampleZQN to MaxZQN"
                mcts_solver.final_criterion = MCTS.MaxZQN(zq=mcts_solver.final_criterion.zq, zn=mcts_solver.final_criterion.zn)
            elseif mcts_solver.final_criterion isa MCTS.SampleZQNS
                @info "Changing final_criterion from SampleZQNS to MaxZQNS"
                mcts_solver.final_criterion = MCTS.MaxZQNS(zq=mcts_solver.final_criterion.zq, zn=mcts_solver.final_criterion.zn)
            end
        end

        if use_bootstrap
            f = mcts_solver.next_action.f
            mcts_solver.init_Q = (bmdp,b,a)->bmdp.belief_reward(bmdp.pomdp, b, a, nothing) + discount(bmdp)*value_lookup(f, @gen(:sp)(bmdp, b, a))
        end

        if !isnothing(depth)
            @info "Changing depth from $(mcts_solver.depth) to $depth"
            mcts_solver.depth = depth
        end

        if !isnothing(k_state) && !isnothing(alpha_state)
            @warn "Adjusting SPW..."
            mcts_solver.enable_state_pw = true
            mcts_solver.k_state = k_state
            mcts_solver.alpha_state = alpha_state
        end

        if !isnothing(k_action) && !isnothing(alpha_action)
            @warn "Adjusting APW..."
            mcts_solver.enable_action_pw = true
            mcts_solver.k_action = k_action
            mcts_solver.alpha_action = alpha_action
        end

        @warn "Adjusting policy/solver..."
    end


    function adjust_nn_params(nn_params; ϵ_greedy=0.0)
        # @warn "Adjusting nn_params..."
        # nn_params.zero_out_tried_actions = true
        # nn_params.use_prioritized_action_selection = false
    end


    function adjust_policy(policy, n_iterations; ϵ_greedy=nothing, zq=nothing, zn=nothing, zqn_argmax=true, k_state=nothing, alpha_state=nothing, k_action=nothing, alpha_action=nothing, depth=nothing)
        policy = deepcopy(policy)
        if !isnothing(n_iterations)
            policy.planner.solver.n_iterations = n_iterations
        end

        adjust_betazero_mcts(policy.planner.solver; zq, zn, zqn_argmax, k_state, alpha_state, k_action, alpha_action, depth)
        adjust_nn_params(policy.planner.next_action.nn_params; ϵ_greedy)

        return policy
    end

    function adjust_solver(solver, n_iterations; ϵ_greedy=nothing, zq=nothing, zn=nothing, zqn_argmax=true, k_state=nothing, alpha_state=nothing, k_action=nothing, alpha_action=nothing, depth=nothing)
        solver = deepcopy(solver)
        if !isnothing(n_iterations)
            solver.mcts_solver.n_iterations = n_iterations
        end

        adjust_betazero_mcts(solver.mcts_solver; zq, zn, zqn_argmax, k_state, alpha_state, k_action, alpha_action, depth)
        adjust_nn_params(solver.nn_params; ϵ_greedy)

        return solver
    end

    function mcts_lavi(mcts_planner, lavi_policy)
        mcts_planner = deepcopy(mcts_planner)
        mcts_planner.solver.estimate_value = (bmdp,b,d)->begin
            b̃ = BetaZero.input_representation(b)
            return value(lavi_policy, b)
        end
        mcts_planner.solver.next_action = (bmdp,b,bnode)->begin
            b̃ = BetaZero.input_representation(b)
            s = convert_s(ParticleHistoryBelief, b̃, bmdp)
            return action(lavi_policy, s)
        end
        mcts_planner = solve(mcts_planner.solver, mcts_planner.mdp) # Important for online policy
        return mcts_planner
    end

    Base.convert(::Type{SVector{1,Float64}}, s::LightDarkState) = SVector{1,Float64}(s.y)
    POMDPs.actionindex(::LightDarkPOMDP, a::Int) = a + 2
    POMDPs.action(p::AlphaVectorPolicy, s::RSState) = action(p, ParticleCollection([s]))
    POMDPs.value(p::AlphaVectorPolicy, s::RSState) = value(p, ParticleCollection([s])) # Bug in https://github.com/LAMDA-POMDP/Test/blob/0f1aefec579a59242060fad9e74683bdaa3b3208/test/RSTest.jl#L10
    AdaOPS.pdf(::Union{Nothing,Float32}, ::Union{Nothing,Float32}) = 1.0 # weight for MinEx nothing or -Inf32 observation

    function solve_adaops(pomdp; use_rocksample_mdp_solution=false, use_fixed_bounds=false)
        timing_results = @elapsed begin
            # From: https://github.com/LAMDA-POMDP/Test/
            if pomdp isa LightDarkPOMDP
                if use_fixed_bounds
                    @warn "Using fixed bound in AdaOPS for LightDark."
                    bds = AdaOPS.IndependentBounds(pomdp.incorrect_r, pomdp.correct_r, check_terminal=true)
                    adaops_solver = AdaOPSSolver(bounds=bds,
                                        m_min=10,
                                        delta=1.0,
                                        rng=Random.GLOBAL_RNG,
                                        tree_in_info=false)
                else
                    interp = LocalGIFunctionApproximator(RectangleGrid(range(-1, stop=1, length=3), range(-100, stop=100, length=401)))
                    approx_mdp = solve(LocalApproximationValueIterationSolver(
                        interp,
                        verbose=true,
                        max_iterations=1000,
                        is_mdp_generative=true,
                        n_generative_samples=1000),
                        pomdp)
                    approx_random = solve(LocalApproximationRandomSolver(
                        interp,
                        verbose=true,
                        max_iterations=1000,
                        is_mdp_generative=true,
                        n_generative_samples=1000),
                        pomdp)
                    grid = StateGrid(range(-10, stop=15, length=26))
                    bds = AdaOPS.IndependentBounds(FOValue(approx_random), FOValue(approx_mdp), check_terminal=true)
                    adaops_solver = AdaOPSSolver(bounds=bds,
                                        grid=grid,
                                        m_min=10,
                                        delta=1.0,
                                        rng=Random.GLOBAL_RNG,
                                        tree_in_info=false)
                end
            elseif pomdp isa RockSamplePOMDP
                move_east = RSExitSolver()
                if use_fixed_bounds
                    @warn "Using fixed bound in AdaOPS for RockSample."
                    n = pomdp.map_size[1]
                    k = length(pomdp.rocks_positions)
                    upper_bnd_fixed = sum(discount(pomdp)^(t-1)*pomdp.good_rock_reward for t in (1+n-k):(2k-n)) + pomdp.exit_reward
                    bds = AdaOPS.IndependentBounds(FOValue(move_east), upper_bnd_fixed, check_terminal=true, consistency_fix_thresh=1e-5)
                else
                    if use_rocksample_mdp_solution
                        @warn "Using MDP solution in AdaOPS for RockSample."
                        upper_bnd_solver = RSMDPSolver()
                    else
                        upper_bnd_solver = RSQMDPSolver()
                    end
                    bds = AdaOPS.IndependentBounds(FOValue(move_east), POValue(upper_bnd_solver), check_terminal=true, consistency_fix_thresh=1e-5)
                end
                adaops_solver = AdaOPSSolver(bounds=bds,
                                    m_min=100,
                                    delta=0.1,
                                    timeout_warning_threshold=Inf,
                                    tree_in_info=false)
            elseif pomdp isa MinExPOMDP
                minex_lb = compute_lowerbound_return_minex(pomdp)
                minex_up = compute_optimal_return_minex(pomdp, initialstate(pomdp))[1] # mean of optimal return
                bds = AdaOPS.IndependentBounds(minex_lb, minex_up, check_terminal=true, consistency_fix_thresh=1e-5)
                bds = init_param(pomdp, bds)
                adaops_solver = AdaOPSSolver(bounds=bds,
                                    m_min=500,
                                    delta=0.1,
                                    timeout_warning_threshold=Inf,
                                    tree_in_info=false)
            else
                error("No AdaOPS planner for $(typeof(pomdp))")
            end
            adaops_planner = solve(adaops_solver, pomdp)
        end
        @info timing_results
        return adaops_planner
    end

    function solve_despot(pomdp; use_rocksample_mdp_solution=false, use_fixed_bounds=false)
        timing_results = @elapsed begin
            if pomdp isa LightDarkPOMDP
                random = solve(RandomSolver(), pomdp)
                if use_fixed_bounds
                    @warn "Using fixed bound in DESPOT for LightDark."
                    bds = ARDESPOT.IndependentBounds(pomdp.incorrect_r, pomdp.correct_r, check_terminal=true)
                    despot_solver = DESPOTSolver(; lambda=0.1, K=30, bounds=bds, bounds_warnings=false)
                else
                    bds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(random), pomdp.correct_r, check_terminal=true)
                    despot_solver = DESPOTSolver(; default_action=random, lambda=0.1, K=30, bounds=bds, bounds_warnings=false)
                end
            elseif pomdp isa RockSamplePOMDP
                move_east = RSExitSolver()
                if use_fixed_bounds
                    @warn "Using fixed bound in DESPOT for RockSample."
                    n = pomdp.map_size[1]
                    k = length(pomdp.rocks_positions)
                    upper_bnd_fixed = sum(discount(pomdp)^(t-1)*pomdp.good_rock_reward for t in (1+n-k):(2k-n)) + pomdp.exit_reward
                    bds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(move_east), upper_bnd_fixed, check_terminal=true, consistency_fix_thresh=1e-5)
                else
                    if use_rocksample_mdp_solution
                        @warn "Using MDP solution in DESPOT for RockSample."
                        upper_bnd_solver = RSMDPSolver()
                    else
                        upper_bnd_solver = RSQMDPSolver()
                    end
                    bds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(move_east), ARDESPOT.FullyObservableValueUB(upper_bnd_solver), check_terminal=true, consistency_fix_thresh=1e-5)
                end
                despot_solver = DESPOTSolver(; default_action=move_east, lambda=0.0, K=100, bounds=bds)
            elseif pomdp isa MinExPOMDP
                random = solve(RandomSolver(), pomdp)
                minex_lb = compute_lowerbound_return_minex(pomdp)
                minex_up = compute_optimal_return_minex(pomdp, initialstate(pomdp))[1] # mean of optimal return
                bds = ARDESPOT.IndependentBounds(minex_lb, minex_up, check_terminal=true, consistency_fix_thresh=1e-5)
                despot_solver = DESPOTSolver(; default_action=random, lambda=0.0, K=100, bounds=bds)
            else
                error("No DESPOT planner for $(typeof(pomdp))")
            end
            despot_planner = solve(despot_solver, pomdp)
        end
        @info timing_results
        return despot_planner
    end
end

depth_sweep = [nothing]
iteration_sweep = [nothing]
max_steps = 100
pomcpow_iteration_factor = 1
n_runs = 100
ϵ_sweep = [0.0]
zq_sweep = [nothing]
zn_sweep = [nothing]
kS_sweep = [nothing]
alphaS_sweep = [nothing]
kA_sweep = [nothing]
alphaA_sweep = [nothing]
skip_z_equal = true
run_with_heuristics = false

ϵ_results = Dict()
z_results = Dict()
spw_results = Dict()
apw_results = Dict()
if !@isdefined(online_results)
    online_results = Dict()
end
for ϵ_greedy in ϵ_sweep
    for zq in zq_sweep
        for zn in zn_sweep
            for k_state in kS_sweep
                for alpha_state in alphaS_sweep
                    for k_action in kA_sweep
                        for alpha_action in alphaA_sweep
                            for depth in depth_sweep
                                ϵ_results[ϵ_greedy] = Dict()
                                if !isnothing(zq) && !isnothing(zn) && haskey(z_results, (zq, zn))
                                    @warn "Skipping ZQN $((zq, zn)), already computed"
                                    continue
                                else
                                    z_results[(zq, zn)] = Dict()
                                end
                                if !isnothing(k_state) && !isnothing(alpha_state) && haskey(spw_results, (k_state, alpha_state))
                                    @warn "Skipping SPW $((k_state, alpha_state)), already computed"
                                    continue
                                else
                                    spw_results[(k_state, alpha_state)] = Dict()
                                end
                                if !isnothing(k_action) && !isnothing(alpha_action) && haskey(apw_results, (k_action, alpha_action))
                                    @warn "Skipping APW $((k_action, alpha_action)), already computed"
                                    continue
                                else
                                    apw_results[(k_action, alpha_action)] = Dict()
                                end
                                for (iteration_i, n_iterations) in enumerate(iteration_sweep)
                                    if skip_z_equal && !isnothing(zq) && !isnothing(zn) && zq == zn && (zq != 1 && zn != 1) && (zq != 0 && zn != 0) # skip equal expect (1,1)
                                        @warn "Skipping zq=$zq and zn=$zn"
                                        continue
                                    end
                                    if @isdefined(policy) && @isdefined(solver)
                                        bz_policy = adjust_policy(policy, n_iterations; ϵ_greedy, zq, zn, k_state, alpha_state, k_action, alpha_action, depth)
                                        bz_solver = adjust_solver(solver, n_iterations; ϵ_greedy, zq, zn, k_state, alpha_state, k_action, alpha_action, depth)
                                    end

                                    @everywhere begin
                                        using LocalApproximationValueIteration
                                        using LocalFunctionApproximation
                                        using GridInterpolations
                                    end

                                    if @isdefined(en)
                                        en_policy = BetaZero.attach_surrogate!(deepcopy(bz_policy), en)
                                    end

                                    # Control seeds for DESPOT and AdaOPS that use a MersenneTwister instead of AbstractRNG types.
                                    Random.seed!(0xC0FFEE)
                                    Random.seed!(rand(1:typemax(UInt32))) # To ensure we mix it up (still deterministic based on previous `seed!` call)

                                    # NOTE: Uncomment the algorithms you want to test.
                                    policies = Dict(
                                        # "BetaZero"=>bz_policy,
                                        # "BetaZero (ensemble)"=>en_policy,
                                        # "Random"=>RandomPolicy(pomdp),
                                        # "MCTS (zeroed values)"=>extract_mcts(bz_solver, pomdp),
                                        # "MCTS (rand. values)"=>extract_mcts_rand_values(bz_solver, pomdp),
                                        # "LAVI"=>lavi_policy,
                                        # "MCTS + LAVI"=>mcts_lavi(bz_policy.planner, lavi_policy),
                                        # "Raw Network [policy]"=>RawNetworkPolicy(pomdp, bz_policy.surrogate),
                                        # "Raw Network [value]"=>RawValueNetworkPolicy(bz_solver.bmdp, bz_policy.surrogate, 5), # NOTE: `n_obs`
                                        # "Raw Network [pfail]"=>RawPfailNetworkPolicy(bz_solver.bmdp, bz_policy.surrogate, 5, bz_solver.mcts_solver.δ), # NOTE: `n_obs`
                                        # "POMCPOW"=>solve_pomcpow(pomdp, pomcpow_iteration_factor*n_iterations, override=true, use_heuristics=run_with_heuristics), # Note override for table comparison.
                                        # "AdaOPS"=>solve_adaops(pomdp; use_rocksample_mdp_solution=false, use_fixed_bounds=!run_with_heuristics), # Note use fixed bound for RS(25,25)
                                        # "DESPOT"=>solve_despot(pomdp; use_rocksample_mdp_solution=false, use_fixed_bounds=!run_with_heuristics), # Note use fixed bound for RS(25,25)
                                    )

                                    latex_table = Dict()
                                    results_table = Dict()
                                    for (k,π) in policies
                                        @info "Baselining $(isnothing(n_iterations) ? solver.mcts_solver.n_iterations : n_iterations) online iterations (over $n_runs runs, with ϵ=$ϵ_greedy, and zq=$zq and zn=$zn, and k_state=$k_state alpha_state=$alpha_state, and k_action=$k_action alpha_action=$alpha_action)"
                                        if iteration_i > 1 && k ∈ ["AdaOPS", "DESPOT", "Raw Network [policy]", "Raw Network [value]", "Random", "LAVI"]
                                            @info "Skipping $k due to not using n_iterations online"
                                            continue
                                        else
                                            @info "Running $k baseline..."
                                        end
                                        n_digits = 2

                                        progress = Progress(n_runs)
                                        channel = RemoteChannel(()->Channel{Bool}(), 1)

                                        @async while take!(channel)
                                            next!(progress)
                                        end

                                        if !haskey(online_results, k)
                                            online_results[k] = (X=[], Y=[], Yerr=[])
                                        end

                                        timing = @timed begin
                                            local results = pmap(i->begin
                                                Random.seed!(i) # Make sure each policy has an apples-to-apples comparison (e.g., same starting episode states, etc.)
                                                Random.seed!(rand(1:typemax(UInt32))) # To ensure that we don't use initial states or beliefs that were seen during BetaZero training (still deterministic based on previous `seed!` call)
                                                ds0 = initialstate(pomdp)
                                                s0 = rand(ds0)
                                                b0 = initialize_belief(up, ds0)
                                                history = simulate(HistoryRecorder(max_steps=max_steps), pomdp, π, up, b0, s0)
                                                g = discounted_reward(history)
                                                R = map(h->h.r, history)
                                                G = BetaZero.compute_returns(R; γ=discount(pomdp))
                                                states = map(h->h.s, history)
                                                push!(states, history[end].sp)
                                                acc = accuracy(pomdp, history[1].b, history[1].s, states, map(h->h.a, history), G)
                                                put!(channel, true) # trigger progress bar update
                                                g, acc
                                            end, 1:n_runs)
                                            returns = vcat([r[1] for r in results]...)
                                            accuracies = vcat([r[2] for r in results]...)
                                            put!(channel, false) # tell printing task to finish
                                            μ, σ = mean_and_std(returns)
                                            μ_rd = round(μ, digits=n_digits)
                                            stderr_rd = round(σ/sqrt(n_runs), digits=n_digits)
                                            μ_acc, σ_acc = mean_and_std(accuracies)
                                            μ_acc_rd = round(μ_acc, digits=n_digits)
                                            stderr_acc_rd = round(σ_acc/sqrt(n_runs), digits=n_digits)
                                        end
                                        time_rd = round(timing.time/n_runs, digits=2n_digits)
                                        @info "$k: $μ_rd ± $stderr_rd [$μ_acc_rd ± $stderr_acc_rd] ($time_rd seconds)"
                                        latex_table[μ_rd] = "$k & \$$μ_rd \\pm $stderr_rd\$ & \$$μ_acc_rd \\pm $stderr_acc_rd\$ & $time_rd s \\\\"
                                        results_table[k] = (returns=(μ_rd, stderr_rd), accs=(μ_acc_rd, stderr_acc_rd), time=time_rd)
                                        push!(online_results[k].X, n_iterations)
                                        push!(online_results[k].Y, μ_rd)
                                        push!(online_results[k].Yerr, stderr_rd)
                                    end

                                    if !isempty(policies)
                                        println("|", "—"^50, "|")
                                        for (k,v) in sort(latex_table, rev=true)
                                            println(v)
                                        end
                                        println("—"^50)
                                        println("    \\item[*] {$n_iterations iterations ($n_runs runs each).}")
                                        println("|", "—"^50, "|")
                                        ϵ_results[ϵ_greedy][n_iterations] = results_table

                                        if haskey(results_table, "BetaZero")
                                            z_results[(zq, zn)][n_iterations] = results_table["BetaZero"]
                                            spw_results[(k_state, alpha_state)][n_iterations] = results_table["BetaZero"]
                                            apw_results[(k_action, alpha_action)][n_iterations] = results_table["BetaZero"]
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end


function plot_epsilon_sweep(ϵ_results)
    plot(size=(600,250))
    n = length(keys(ϵ_results))
    cmap = cgrad(:viridis, n)
    for (i,ϵ) in enumerate(sort(collect(keys(ϵ_results))))
        x = []
        y = []
        yerr = []
        for iters in sort(collect(keys(ϵ_results[ϵ])))
            μ, σ = ϵ_results[ϵ][iters]["BetaZero"].returns
            push!(x, iters)
            push!(y, μ)
            push!(yerr, σ)
        end
        c = get(cmap, i/n)
        plot!(x, y, yerror=yerr, lw=2, label=ϵ, fillalpha=0.2, msc=c, c=c)
    end
    return plot!()
end

nothing # REPL