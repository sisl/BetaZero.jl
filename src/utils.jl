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
