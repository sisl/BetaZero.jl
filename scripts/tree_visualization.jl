using D3Trees
using Statistics
using StatsBase

showtree(tree) = inbrowser(D3Tree(tree), Sys.islinux() ? "firefox" : "google-chrome")

function BetaZero.MCTS.node_tag(b)
    return "belief [$(hash(b))]"
end

function BetaZero.MCTS.node_tag(a::Union{Tuple,Symbol,Int})
    return "$a"
end
