"""
Run @time on expression based on `verbose` flag.
"""
macro conditional_time(verbose, expr)
    esc(quote
        if $verbose
            @time $expr
        else
            $expr
        end
    end)
end


"""
Return the size of the belief representation for a given POMDP (used as input to surrogates).
"""
function get_input_size(pomdp::POMDP, up::Updater)
    ds0 = initialstate(pomdp)
    b0 = initialize_belief(up, ds0)
    b̃ = input_representation(b0)
    return size(b̃)
end


"""
Save figures as PNG with higher (or custom) dpi.
"""
bettersavefig(filename; kwargs...) = bettersavefig(plot!(), filename; kwargs...)
function bettersavefig(fig, filename; dpi=300)
    if length(filename) >= 4 && filename[end-3:end] == ".png"
        filename = filename[1:end-4]
    end
    filename_png, filename_svg = "$filename.png", "$filename.svg"
    Plots.savefig(fig, filename_svg)
    if Sys.iswindows()
        run(`inkscape -f $filename_svg -e $filename_png -d $dpi`)
    else
        run(`inkscape $filename_svg -o $filename_png -d $dpi`)
    end
    rm(filename_svg)
end


normalize01(x, X) = (x - minimum(X)) / (maximum(X) - minimum(X))


"""
Return colormap where 0 is at the center (if `xmin` is above zero, then anchor at 0)
"""
function shifted_colormap(xmin, xmax; colors=[:red, :white, :green], rev=false)
	if xmin ≥ 0
		buckets = [0, xmin, xmin/2, xmax/2, xmax] # only non-negatives, anchor at 0
		colors = colors[2:end]
	else
		buckets = [xmin, xmin/2, 0, xmax/2, xmax] # shift colormap so 0 is at center
	end
    normed = (buckets .- xmin) / (xmax - xmin)
    return cgrad(colors, normed, rev=rev)
end

function shifted_colormap(X; kwargs...)
	xmin, xmax = minimum(X), maximum(X)
	return shifted_colormap(xmin, xmax; kwargs...)
end


"""
Calculate total number of simulations ran during BetaZero offline training.
"""
count_simulations(solver::BetaZeroSolver) = solver.params.n_iterations * (solver.params.n_data_gen + 2*solver.params.n_evaluate)
count_simulations_accumulated(solver::BetaZeroSolver; zero_start::Bool=true, init_i=zero_start ? 0 : 1) = [i * (solver.params.n_data_gen + 2*solver.params.n_evaluate) for i in init_i:solver.params.n_iterations]


"""
Calculate mean and standard error = std/sqrt(n).
"""
function mean_and_stderr(X)
    μ, σ = mean_and_std(X)
    n = length(X)
    return μ, σ/sqrt(n)
end


"""
Return the sequence from `Y` with the latest max up to that point.
"""
function rolling_max(Y)
	max_Y = Vector{Float64}(undef, length(Y))
	for i in eachindex(Y)
		if i == 1
			max_Y[i] = Y[i]
		else
			max_Y[i] = max(Y[i], maximum(max_Y[1:i-1]))
		end
	end
    return max_Y
end


"""
Compute rolling max of `Y` and match the indices of `Y2`.
"""
function rolling_max(Y, Y2)
	max_Y = Vector{Float64}(undef, length(Y))
    max_Y2 = Vector{Float64}(undef, length(Y2))
	for i in eachindex(Y)
		if i == 1
			max_Y[i] = Y[i]
            max_Y2[i] = Y2[i]
		else
			prev_max_idx = argmax(max_Y[1:i-1])
			prev_max = max_Y[prev_max_idx]
			curr_max_idx = Y[i] > prev_max ? i : prev_max_idx
			max_Y[i] = Y[curr_max_idx]
            max_Y2[i] = Y2[curr_max_idx]
		end
	end
    return max_Y, max_Y2
end


rolling_sum(X) = [sum(X[1:i]) for i in eachindex(X)]
rolling_mean(X) = [mean(X[1:i]) for i in eachindex(X)]
rolling_mean(X, window) = [mean(X[(i - window < 1 ? 1 : i - window):(i + window > length(X) ? length(X) : i + window)]) for i in eachindex(X)]
rolling_stderr(X) = [std(X[1:i])/sqrt(i) for i in eachindex(X)]
rolling_stderr(X, window) = [begin X′ = X[(i - window < 1 ? 1 : i - window):(i + window > length(X) ? length(X) : i + window)]; std(X′)/sqrt(length(X′)) end for i in eachindex(X)]


# Exponential Smoothing (from Crux.jl)
function smooth(v, weight = 0.6)
    N = length(v)
    smoothed = Array{Float64, 1}(undef, N)
    smoothed[1] = v[1]
    for i = 2:N
        smoothed[i] = smoothed[i-1] * weight + (1 - weight) * v[i]
    end
    return smoothed
end


function dirichlet_noise(p; α=1, ϵ=0.25)
    k = length(p)
    η = rand(Dirichlet(k, α))
    return (1 - ϵ)*p + ϵ*η
end


"""
Return initial Q function to bootstrap the search.
"""
function bootstrap(f)
    return (bmdp,b,a)->bmdp.belief_reward(bmdp.pomdp, b, a, nothing) + discount(bmdp)*value_lookup(f, @gen(:sp)(bmdp, b, a))
end


"""
Install supporting example POMDP models, the `RemoteJobs` package, and the `ParticleBeliefs` wrapper.
"""
function install_extras()
    @info "Installing POMDP models and tools..."
    packages = [
        PackageSpec(url=joinpath(@__DIR__, "..", "submodules", "LightDark")),
        PackageSpec(url="https://github.com/sisl/MineralExploration"),
        PackageSpec(url=joinpath(@__DIR__, "..", "submodules", "MinEx")),
        PackageSpec(url=joinpath(@__DIR__, "..", "submodules", "RemoteJobs")),
        PackageSpec(url=joinpath(@__DIR__, "..", "submodules", "ParticleBeliefs")),
    ]

    # Run dev altogether
    # This is important that it's run together so there
    # are no "expected pacakge X to be registered" errors.
    Pkg.develop(packages)
end
