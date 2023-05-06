m = 5
en = BetaZero.EnsembleNetwork(networks=[deepcopy(policy.surrogate) for _ in 1:m])

solver2 = deepcopy(solver)
solver2.nn_params.verbose_plot_frequency = Inf
en = BetaZero.train_ensemble(solver2, en; verbose=true)

μ, σ = en([-10 0]')
