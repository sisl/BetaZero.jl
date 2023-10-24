using Distributed
@everywhere begin
    using Hyperopt
    using Parameters
    using Optim
    using BetaZero
    using LinearAlgebra
    using POMCPOW
    using Random
    using BSON

    include("representation_minex.jl")

    solver = POMCPOWSolver(
        estimate_value=0,
        criterion=POMCPOW.MaxUCB(100.0),
        tree_queries=10_000,
        k_action=4.0,
        alpha_action=0.5,
        k_observation=2.0,
        alpha_observation=0.25,
        tree_in_info=false)

    function simulated_return(solver::Union{PUCTSolver,DPWSolver}, pomdp::POMDP, up::Updater, belief_reward::Function, exploration_constant::Float64, k_action::Real, alpha_action::Real, k_state::Real, alpha_state::Real; n::Int=100, seed=0, λ_lcb=0)
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
            simulate(RolloutSimulator(), pomdp, planner, up, b0, s0)
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
    -returns # Note: Negative so as to maximize returns
end
