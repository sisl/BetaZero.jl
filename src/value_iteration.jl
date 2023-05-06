using ParticleBeliefs
using LocalApproximationValueIteration
using LocalFunctionApproximation
using GridInterpolations

# Discretization of [μ, σ] belief representation
min_mean, max_mean = -30, 30
min_sigma, max_sigma = 0, 5

discrete_length = 100

grid = RectangleGrid(range(min_mean, stop=max_mean, length=discrete_length),
	                 range(min_sigma, stop=max_sigma, length=discrete_length))

interpolation = LocalGIFunctionApproximator(grid)

vi_solver = LocalApproximationValueIterationSolver(interpolation,
											       max_iterations=100,
	        	  	 						       is_mdp_generative=true,
                                                   verbose=true,
												   n_generative_samples=10)
bmdp = BetaZero.fill_bmdp!(pomdp, solver)
# @time lavi_policy = solve(vi_solver, bmdp)

function plot_value_function(policy) # TODO: value_plot
	plot()
	title!("value function")
	xlabel!("belief std")
	ylabel!("belief mean")
    
    cmap = cgrad([:white, :green])
	heatmap!(grid.cutPoints[2], grid.cutPoints[1], (σ,μ)->value(policy, [μ,σ]), fill=true, c=cmap, cbar=true) # NOTE: x-y flip
end

plot_value_function(lavi_policy) |> display

a2b = (μ,σ)->convert_s(ParticleHistoryBelief, [μ,σ], bmdp)

function plot_policy(policy, A=nothing) # TODO: policy_plot
	plot()
	title!("policy")
	xlabel!("belief std")
	ylabel!("belief mean")

    policy_palette = palette(:viridis, 3)

    if isnothing(A)
        A = (σ,μ)->action(policy, a2b(μ,σ)) # NOTE: x-y flip
    end

    heatmap!(grid.cutPoints[2], grid.cutPoints[1], A, color=policy_palette)
end

@time A = [action(lavi_policy, a2b(μ,σ)) for μ in grid.cutPoints[1], σ in grid.cutPoints[2]]
plot_policy(lavi_policy, A) |> display


plot(plot_value_function(lavi_policy), plot_policy(lavi_policy, A), layout=2, size=(1100,300), margin=5Plots.mm) |> display
# BetaZero.bettersavefig("value_and_policy_plots_lightdark")