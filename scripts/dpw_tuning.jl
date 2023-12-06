using Distributed
@everywhere begin
    using Hyperopt
    using Parameters
    using Optim
    using BetaZero
    using LinearAlgebra
    using POMDPTools
    using Random

    include("representation_rocksample.jl")

    uniform_policy_vector = normalize(ones(length(actions(pomdp))), 1)

    solver = PUCTSolver(n_iterations=50,
        exploration_constant=50.0,
        depth=10, # 5
        k_action=10.0,
        alpha_action=0.5,
        k_state=2.0,
        alpha_state=0.1,
        estimate_value=(bmdp,b,d)->0,
        estimate_policy=(bmdp,b)->uniform_policy_vector,
    )

    USE_NETWORK = true

    if USE_NETWORK
        network = load_policy(joinpath(@__DIR__, "..", "..", "policy_rocksample")).surrogate
        nn_params = BetaZeroNetworkParameters(input_size=BetaZero.get_input_size(pomdp,up), action_size=length(actions(pomdp)))
        solver.estimate_value=(bmdp,b,d)->BetaZero.value_lookup(network, b)
        solver.estimate_policy=(bmdp,b)->BetaZero.policy_lookup(network, b)
        solver.next_action = (bmdp,b,bnode)->BetaZero.next_action(bmdp, b, network, nn_params, bnode)
    end

    function simulated_return(solver::Union{PUCTSolver,DPWSolver}, pomdp::POMDP, up::Updater, belief_reward::Function, exploration_constant::Float64, k_action::Real, alpha_action::Real, k_state::Real, alpha_state::Real; n::Int=100, seed=0, λ_lcb=0, max_steps=100)
        solver = deepcopy(solver)
        solver.exploration_constant = exploration_constant
        solver.k_action = k_action
        solver.alpha_action = alpha_action
        solver.k_state = k_state
        solver.alpha_state = alpha_state

        bmdp = BeliefMDP(pomdp, up, belief_reward)
        planner = solve(solver, bmdp)

        μ, σ = mean_and_std([begin
            Random.seed!(seed+i) # Seed to control initial state
            ds0 = initialstate(pomdp)
            s0 = rand(ds0)
            b0 = initialize_belief(up, ds0)
            ret = simulate(RolloutSimulator(; max_steps), pomdp, planner, up, b0, s0)
            ret
        end for i in 1:n])

        μ - λ_lcb*σ # lower confidence bound
    end
end

N = 100
horp = @phyperopt for i=N, sampler=LHSampler(),
        exploration_constant=LinRange(1,1000,N),
        k_action=LinRange(1,100,N),
        alpha_action=LinRange(0,1,N),
        k_state=LinRange(1,100,N),
        alpha_state=LinRange(0,1,N)
    @info "Start $i..."
    belief_reward = rocksample_belief_reward
    returns = simulated_return(solver, pomdp, up, belief_reward, exploration_constant, k_action, alpha_action, k_state, alpha_state; λ_lcb=0)
    @show i, exploration_constant, k_action, alpha_action, k_state, alpha_state, returns
    # Negative so as to maximize returns
    -returns
end
