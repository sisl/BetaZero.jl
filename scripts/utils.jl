bettersavefig(filename; kwargs...) = bettersavefig(plot!(), filename; kwargs...)
function bettersavefig(fig, filename; dpi=300)
    filename_png, filename_svg = "$filename.png", "$filename.svg"
    savefig(fig, filename_svg)
    run(`inkscape -f $filename_svg -e $filename_png -d $dpi`)
    rm(filename_svg)
end
