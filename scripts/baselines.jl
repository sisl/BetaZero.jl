# Baseline using POMCOW
# Baseline using MCTS w/out network
# Baseline using one-step lookahead
# Baseline using random policy
using POMCPOW
using ProgressMeter
using Distributed

REINCLUDE = false
if REINCLUDE
    include("lightdark.jl")
    policy = BetaZero.BSON.@load "policy_100iters_lightdark.bson"
    solver = BetaZero.BSON.@load "solver_100iters_lightdark.bson"
end

function solve_osla(f, pomdp, up, belief_reward; n_actions=10, n_obs=10)
    @show n_actions, n_obs
    solver = OneStepLookaheadSolver(n_actions=n_actions, n_obs=n_obs)
    solver.estimate_value = b->BetaZero.value_lookup(b, f)
    bmdp = BeliefMDP(pomdp, up, belief_reward)
    planner = solve(solver, bmdp)
    return planner
end

function extract_mcts(solver, pomdp)
    mcts_solver = deepcopy(solver.mcts_solver)
    mcts_solver.estimate_value = (bmdp,b,d)->0.0
    planner = solve(mcts_solver, solver.bmdp)
    return planner
end

function extract_mcts_rand_values(solver, pomdp)
    mcts_solver = deepcopy(solver.mcts_solver)
    mcts_solver.estimate_value = (bmdp,b,d)->rand()
    planner = solve(mcts_solver, solver.bmdp)
    return planner
end

function convert_to_pomcow(solver::BetaZeroSolver)
    return POMCPOWSolver(tree_queries=solver.mcts_solver.n_iterations,
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

function adjust_solver(solver, n_iterations)
    policy = deepcopy(policy)
    policy.planner.solver.n_iterations = n_iterations
    return policy
end

iteration_sweep = [10, 100, 1000, 10_000]
i_iteration = 4
n_iterations = iteration_sweep[i_iteration]
solver.mcts_solver.n_iterations = n_iterations # linked to `policy.planner.solver`

# action_obs_scale = 5
# osla_n_actions = log(10,n_iterations)*i_iteration
# osla_n_obs = log(10,n_iterations)^(i_iteration-1)
osla_n_actions = n_iterations
osla_n_obs = 1

policies = Dict(
    "BetaZero"=>policy,
    "Random"=>RandomPolicy(pomdp),
    "One-Step Lookahead"=>solve_osla(policy.network, pomdp, up, lightdark_belief_reward; n_actions=osla_n_actions, n_obs=osla_n_obs),
    # "MCTS (zeroed values)"=>extract_mcts(solver, pomdp),
    # "MCTS (rand. values)"=>extract_mcts_rand_values(solver, pomdp),
    "POMCPOW"=>solve(convert_to_pomcow(solver), pomdp)
)

n_runs = 1000
latex_table = Dict()
for (k,π) in policies
    global latex_table
    @info "Running $k baseline..."
    n_digits = 3

    progress = Progress(n_runs)
    channel = RemoteChannel(()->Channel{Bool}(), 1)

    @async while take!(channel)
        next!(progress)
    end

    timing = @timed begin
        local returns = pmap(i->begin
            # @show i
            G = simulate(RolloutSimulator(max_steps=200), pomdp, π, up)
            put!(channel, true) # trigger progress bar update
            G
        end, 1:n_runs)
        put!(channel, false) # tell printing task to finish
        μ, σ = mean_and_std(returns)
        μ_rd = round(μ, digits=n_digits)
        stderr_rd = round(σ/sqrt(n_runs), digits=n_digits)
    end
    time_rd = round(timing.time, digits=n_digits)
    @info "$k: $μ_rd ± $stderr_rd ($time_rd seconds)"
    latex_table[μ_rd] = "$k & \$$μ_rd \\pm $stderr_rd\$ & $time_rd s \\\\"
end

for (k,v) in sort(latex_table, rev=true)
    println(v)
end
println("    \\item[*] {$n_iterations iterations ($n_runs runs each).}")
