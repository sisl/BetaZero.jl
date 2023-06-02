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


@with_kw struct SampleN
    τ = 1
end

"""
Return the best action based on exponentiated visit counts.
"""
function select_best(crit::SampleN, tree, snode::Int)
    _, N = compute_qvalues_and_counts(tree, snode)
    τ = crit.τ
    A = tree.children[snode]
    P = N.^(1/τ) ./ sum(N.^(1/τ)) # exponentiated visit counts
    return rand(SparseCat(A, P))
end


struct MaxQN end

function compute_qvalues_and_counts(tree, snode::Int)
    m = length(tree.children[snode])
    Q = Vector{Float64}(undef, m)
    N = Vector{Float64}(undef, m)
    for (i,child) in enumerate(tree.children[snode])
        Q[i] = tree.q[child]
        N[i] = tree.n[child]
    end
    return Q, N
end

"""
Return the best action using information from both Q-values and visit counts.
"""
function select_best(::MaxQN, tree, snode::Int)
    Q, N = compute_qvalues_and_counts(tree, snode)

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


@with_kw struct SampleQN
    τ = 1
end

"""
Return the best action using information from both Q-values and visit counts.
"""
function select_best(crit::SampleQN, tree, snode::Int)
    τ = crit.τ
    if τ == 0
        return select_best(MaxQN(), tree, snode)
    else
        Q, N = compute_qvalues_and_counts(tree, snode)
        QN = (softmax(Q) .* (N ./ sum(N))) .^ (1/τ)
        P = normalize(QN, 1)
        A = tree.children[snode]
        return rand(SparseCat(A, P))
    end
end


@with_kw struct MaxWeightedQN
    w = 0.5
end


"""
Return the best action using information from both Q-values and visit counts.
"""
function select_best(crit::MaxWeightedQN, tree, snode::Int)
    Q, N = compute_qvalues_and_counts(tree, snode)
    w = crit.w

    best_QN = -Inf
    sanode = 0
    QN = w*softmax(Q) .+ (1-w)*(N ./ sum(N))
    for (i,child) in enumerate(tree.children[snode])
        if QN[i] > best_QN
            best_QN = QN[i]
            sanode = child
        end
    end
    return sanode
end



@with_kw struct SampleWeightedQN
    τ = 1
    w = 0.5
end

"""
Return the best action using information from both Q-values and visit counts.
"""
function select_best(crit::SampleWeightedQN, tree, snode::Int)
    τ = crit.τ
    w = crit.w
    if τ == 0
        return select_best(MaxWeightedQN(crit.w), tree, snode)
    else
        Q, N = compute_qvalues_and_counts(tree, snode)
        QN = (w*softmax(Q) .+ (1-w)*(N ./ sum(N))) .^ (1/τ)
        P = normalize(QN, 1)
        A = tree.children[snode]
        return rand(SparseCat(A, P))
    end
end





@with_kw struct MaxZQN
    zq = 1
    zn = 1
end


"""
Return the best action using information from both Q-values and visit counts.
"""
function select_best(crit::MaxZQN, tree, snode::Int)
    Q, N = compute_qvalues_and_counts(tree, snode)
    zq = crit.zq
    zn = crit.zn
    best_QN = -Inf
    sanode = 0
    Nnorm = (N ./ sum(N))
    QN = (softmax(Q).^zq .* Nnorm.^zn) # maximization doesn't need normalization
    for (i,child) in enumerate(tree.children[snode])
        if QN[i] > best_QN
            best_QN = QN[i]
            sanode = child
        end
    end
    return sanode
end



@with_kw mutable struct SampleZQN
    τ = 1
    zq = 1
    zn = 1
end

"""
Return the best action using information from both Q-values and visit counts.
"""
function select_best(crit::SampleZQN, tree, snode::Int)
    τ = crit.τ
    zq = crit.zq
    zn = crit.zn
    if τ == 0
        return select_best(MaxZQN(zq=crit.zq, zn=crit.zn), tree, snode)
    else
        Q, N = compute_qvalues_and_counts(tree, snode)
        Nnorm = (N ./ sum(N))
        QN = (softmax(Q).^zq .* Nnorm.^zn) .^ (1/τ)
        P = normalize(QN, 1)
        A = tree.children[snode]
        return rand(SparseCat(A, P))
    end
end

