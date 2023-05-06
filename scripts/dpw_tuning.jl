using Distributed
@everywhere begin
    using Hyperopt
    using Parameters
    using Optim
    using BetaZero
    using LinearAlgebra
    using Random

    include("representation_minex.jl")
    network = load_incremental("betazero_policy_2_on_par_pomcpow.bson").network

    uniform_policy_vector = normalize(ones(length(actions(pomdp))), 1)

    solver = PUCTSolver(n_iterations=50,
        exploration_constant=50.0,
        depth=5,
        k_action=10.0,
        alpha_action=0.5,
        k_state=2.0,
        alpha_state=0.1,
        estimate_value=(bmdp,b,d)->0,
        estimate_policy=(bmdp,b)->uniform_policy_vector,
    )

    nn_params = BetaZeroNetworkParameters(input_size=BetaZero.get_input_size(pomdp,up), action_size=length(actions(pomdp)))
    solver.estimate_value=(bmdp,b,d)->BetaZero.value_lookup(network, b)
    solver.estimate_policy=(bmdp,b)->BetaZero.policy_lookup(network, b)
    solver.next_action = (bmdp,b,bnode)->BetaZero.next_action(bmdp, b, network, nn_params, bnode)

    function simulated_return(solver::Union{PUCTSolver,DPWSolver}, pomdp::POMDP, up::Updater, belief_reward::Function, exploration_constant::Float64, k_action::Real, alpha_action::Real, k_state::Real, alpha_state::Real; n::Int=100, seed=0, λ_lcb=0)
        solver = deepcopy(solver)
        solver.exploration_constant = exploration_constant
        solver.k_action = k_action
        solver.alpha_action = alpha_action
        solver.k_state = k_state
        solver.alpha_state = alpha_state

        bmdp = BeliefMDP(pomdp, up, belief_reward)
        planner = solve(solver, bmdp)

        # Random.seed!(seed)
        μ, σ = mean_and_std([begin
            Random.seed!(seed+i) # Seed to control initial state
            ds0 = initialstate_distribution(pomdp)
            s0 = rand(ds0)
            b0 = initialize_belief(up, ds0)

            # Random.seed!(10000*(seed+i)) # Seed to randomize MCTS search
            ret = simulate(RolloutSimulator(), pomdp, planner, up, b0, s0)
            # @info ret
            ret
        end for i in 1:n])

        μ - λ_lcb*σ # lower confidence bound
    end
end

N = 100
horp = @phyperopt for i=N, sampler=LHSampler(), 
        exploration_constant=LinRange(1,100,N),
        k_action=LinRange(1,50,N),
        alpha_action=LinRange(0,1,N),
        k_state=LinRange(1,50,N),
        alpha_state=LinRange(0,1,N)
    @info "Start $i..."
    returns = simulated_return(solver, pomdp, up, simple_minex_belief_reward, exploration_constant, k_action, alpha_action, k_state, alpha_state; λ_lcb=1)
    @show i, exploration_constant, k_action, alpha_action, k_state, alpha_state, returns
    # Negative so as to maximize returns
    -returns
end
