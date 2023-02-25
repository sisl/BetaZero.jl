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
