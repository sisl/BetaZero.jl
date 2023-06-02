using Distributed
addprocs([("user@hostname1", 25), ("user@hostname2", 25)], tunnel=true) # launch 50 processes across two separate hosts

@everywhere begin
    using BetaZero
    using LightDark

    pomdp = LightDarkPOMDP()
    up = BootstrapFilter(pomdp, 500)

    function BetaZero.input_representation(b::ParticleCollection{LightDarkState})
        # Function to get belief representation as input to neural network.
        μ, σ = mean_and_std(s.y for s in particles(b))
        return Float32[μ, σ]
    end

    function BetaZero.accuracy(pomdp::LightDarkPOMDP, b0, s0, states, actions, returns)
        # Function to determine accuracy of agent's final decision.
        return returns[end] == pomdp.correct_r
    end
end

solver = BetaZeroSolver(pomdp=pomdp,
                        updater=up,
                        params=BetaZeroParameters(
                            n_iterations=50,
                            n_data_gen=500, # Note increased to 500 when running in parallel.
                        ),
                        nn_params=BetaZeroNetworkParameters(
                            pomdp, up;
                            training_epochs=50,
                            n_samples=100_000,
                            batchsize=1024,
                            learning_rate=1e-4,
                            λ_regularization=1e-5,
                            use_dropout=true,
                            p_dropout=0.2,
                        ),
                        verbose=true,
                        collect_metrics=true,
                        plot_incremental_data_gen=true)

policy = solve(solver, pomdp)
save_policy(policy, "policy.bson")
save_solver(solver, "solver.bson")
