using ParticleBeliefs
using LocalApproximationValueIteration
using LocalFunctionApproximation
using GridInterpolations
using BSON

RESOLVE = true
ENV["LAUNCH_PARALLEL"] = false
include("launch_lightdark.jl")

# Discretization of [μ, σ] belief representation
min_mean, max_mean = -3, 12
min_sigma, max_sigma = 0, 5

discrete_length = 25

discrete_grid = RectangleGrid(range(min_mean, stop=max_mean, length=discrete_length),
                              range(min_sigma, stop=max_sigma, length=discrete_length))

if RESOLVE
	timing_results = @elapsed begin
		interpolation = LocalGIFunctionApproximator(discrete_grid)
		vi_solver = LocalApproximationValueIterationSolver(interpolation,
														max_iterations=25,
														is_mdp_generative=true,
														verbose=true,
														n_generative_samples=100) # 10
		bmdp = BeliefMDP(pomdp, up, lightdark_belief_reward)
		@time lavi_policy = solve(vi_solver, bmdp)
		BSON.@save "policy_lavi_ld10_timing.bson" lavi_policy
		@info "Running statistics..."
		n_sims = 100
		@show mean_and_stderr(simulate(RolloutSimulator(max_steps=100), bmdp, lavi_policy) for _ in 1:n_sims)
	end
	@info timing_results
end


function plot_value_function(policy) # TODO: value_plot
	plot()
	Plots.title!("value function")
	Plots.xlabel!("belief std")
	Plots.ylabel!("belief mean")

    cmap = cgrad([:white, :green])
	heatmap!(discrete_grid.cutPoints[2], discrete_grid.cutPoints[1], (σ,μ)->value(policy, [μ,σ]), fill=true, c=cmap, cbar=true) # NOTE: x-y flip
end

plot_value_function(lavi_policy) |> display

a2b = (μ,σ)->convert_s(ParticleHistoryBelief{LightDarkState}, [μ,σ], bmdp)

function plot_policy(policy, A=nothing) # TODO: policy_plot
	plot()
	Plots.title!("policy")
	Plots.xlabel!("belief std")
	Plots.ylabel!("belief mean")

    policy_palette = palette(:viridis, 3)

    if isnothing(A)
        A = (σ,μ)->action(policy, a2b(μ,σ)) # NOTE: x-y flip
    end

    heatmap!(discrete_grid.cutPoints[2], discrete_grid.cutPoints[1], A, color=policy_palette)
end

# @time A = [action(lavi_policy, a2b(μ,σ)) for μ in discrete_grid.cutPoints[1], σ in discrete_grid.cutPoints[2]]
# plot_policy(lavi_policy, A) |> display

# plot(plot_value_function(lavi_policy), plot_policy(lavi_policy, A), layout=2, size=(1100,300), margin=5Plots.mm) |> display
# BetaZero.bettersavefig("value_and_policy_plots_lightdark")