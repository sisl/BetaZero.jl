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
