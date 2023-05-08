struct MaxQ end

"""
Return the best action based on the maximum Q-value.
"""
function select_best(::MaxQ, tree, snode::Int)
    best_Q = -Inf
    sanode = 0
    for child in tree.children[snode]
        if tree.q[child] > best_Q
            best_Q = tree.q[child]
            sanode = child
        end
    end
    return sanode
end


struct MaxN end

"""
Return the best action based on the maximum visit count.
"""
function select_best(::MaxN, tree, snode::Int)
    best_N = -Inf
    sanode = 0
    for child in tree.children[snode]
        if tree.n[child] > best_N
            best_N = tree.n[child]
            sanode = child
        end
    end
    return sanode
end


struct MaxQN end

"""
Return the best action using information from both Q-values and visit counts.
"""
function select_best(::MaxQN, tree, snode::Int)
    m = length(tree.children[snode])
    Q = Vector{Float64}(undef, m)
    N = Vector{Float64}(undef, m)
    for (i,child) in enumerate(tree.children[snode])
        Q[i] = tree.q[child]
        N[i] = tree.n[child]
    end

    best_QN = -Inf
    sanode = 0
    QN = softmax(Q) .* (N ./ sum(N)) # no normalization necessary for max.
    for (i,child) in enumerate(tree.children[snode])
        if QN[i] > best_QN
            best_QN = QN[i]
            sanode = child
        end
    end
    return sanode
end
