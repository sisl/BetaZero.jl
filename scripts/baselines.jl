using Distributed
@everywhere FROM_PARALLEL_RUN = true

if !FROM_PARALLEL_RUN
    desired_procs = 10
    nprocs() < desired_procs && addprocs(desired_procs)
end


@everywhere begin
    using POMCPOW
    using ProgressMeter
    using BetaZero

    if !FROM_PARALLEL_RUN && !@isdefined(en)
        include("lightdark.jl")
        using BetaZero.MCTS
        using BetaZero.GaussianProcesses
        using BetaZero.DataStructures
        policy = BetaZero.load_policy(joinpath(@__DIR__, "..", "..", "data", "policy_lightdark_64relu.bson"))
        solver = BetaZero.load_solver(joinpath(@__DIR__, "..", "data", "solver_lightdark_64relu.bson"))
        # en = BetaZero.BSON.load(joinpath(@__DIR__, "..", "..", "data", "ensemble_m5_weekend_50iters_include_missing.bson"))[:en]
    end

    function solve_osla(f, pomdp, up, belief_reward, next_action=nothing; n_actions=10, n_obs=10)
        @show n_actions, n_obs
        solver = OneStepLookaheadSolver(n_actions=n_actions, n_obs=n_obs)
        solver.estimate_value = b->BetaZero.value_lookup(b, f)
        solver.next_action = next_action
        bmdp = BeliefMDP(pomdp, up, belief_reward)
        planner = solve(solver, bmdp)
        return planner
    end

    function extract_mcts(solver, pomdp)
        mcts_solver = deepcopy(solver.mcts_solver)
        mcts_solver.estimate_value = (bmdp,b,d)->0.0
        mcts_solver.next_action = RandomActionGenerator()
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

    function convert_to_pomcow(solver::BetaZeroSolver, n_iterations=solver.mcts_solver.n_iterations)
        return POMCPOWSolver(tree_queries=n_iterations,
                            check_repeat_obs=true,
                            check_repeat_act=true,
                            k_action=solver.mcts_solver.k_action,
                            alpha_action=solver.mcts_solver.alpha_action,
                            k_observation=2.0,
                            alpha_observation=0.1,
                            criterion=POMCPOW.MaxUCB(solver.mcts_solver.exploration_constant), # 90 in paper (using discrete states)
                            final_criterion=POMCPOW.MaxQ(),
                            estimate_value=0.0,
                            max_depth=solver.mcts_solver.depth)
    end

    function adjust_policy(policy, n_iterations)
        policy = deepcopy(policy)
        policy.planner.solver.n_iterations = n_iterations
        return policy
    end

    function adjust_solver(solver, n_iterations)
        solver = deepcopy(solver)
        solver.mcts_solver.n_iterations = n_iterations
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
end


# iteration_sweep = [10, 100, 1000, 10_000]
iteration_sweep = [100]

for n_iterations in iteration_sweep
    @info "Baselining $n_iterations online iterations"
    bz_policy = adjust_policy(policy, n_iterations)
    bz_solver = adjust_solver(solver, n_iterations)

    osla_n_actions = n_iterations
    osla_n_obs = 1

    @everywhere begin
        using LocalApproximationValueIteration
        using LocalFunctionApproximation
        using GridInterpolations
    end

    if @isdefined(en)
        en_policy = BetaZero.attach_surrogate!(deepcopy(bz_policy), en)
    end

    policies = Dict(
        "BetaZero"=>bz_policy,
        # "BetaZero (ensemble)"=>en_policy,
        "Random"=>RandomPolicy(pomdp),
        # "One-Step Lookahead"=>solve_osla(bz_policy.surrogate, pomdp, up, lightdark_belief_reward, bz_policy.planner.next_action; n_actions=osla_n_actions, n_obs=osla_n_obs),
        # "One-Step Lookahead (ensemble)"=>solve_osla(en_policy.surrogate, pomdp, up, lightdark_belief_reward, en_policy.planner.next_action; n_actions=osla_n_actions, n_obs=osla_n_obs),
        # "MCTS (zeroed values)"=>extract_mcts(bz_solver, pomdp),
        # "MCTS (rand. values)"=>extract_mcts_rand_values(bz_solver, pomdp),
        "POMCPOW"=>solve(convert_to_pomcow(bz_solver, 100*bz_solver.mcts_solver.n_iterations), pomdp),
        "LAVI"=>lavi_policy,
        "MCTS + LAVI"=>mcts_lavi(bz_policy.planner, lavi_policy),
        "Raw Network"=>RawNetworkPolicy(pomdp, bz_policy.surrogate),
    )

    n_runs = 100
    latex_table = Dict()
    for (k,π) in policies
        @info "Running $k baseline..."
        n_digits = 3

        progress = Progress(n_runs)
        channel = RemoteChannel(()->Channel{Bool}(), 1)

        @async while take!(channel)
            next!(progress)
        end

        timing = @timed begin
            local results = pmap(i->begin
                Random.seed!(i) # Make sure each policy has an apples-to-apples comparison (e.g., same starting episode states, etc.)
                Random.seed!(rand(1:typemax(UInt64))) # To ensure that we don't use initial states or beliefs that were seen during BetaZero training (still deterministic based on previous `seed!` call)
                G = simulate(RolloutSimulator(max_steps=100), pomdp, π, up)
                accuracy = G > 0.0
                put!(channel, true) # trigger progress bar update
                G, accuracy
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
    end

    for (k,v) in sort(latex_table, rev=true)
        println(v)
    end
    println("—"^50)
    println("    \\item[*] {$n_iterations iterations ($n_runs runs each).}")
    println("—"^50)
end