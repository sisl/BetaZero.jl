abstract type Criteria end


"""
Extract either `qvalues` (Q), `counts` (N), or `safety` (S) from the tree.
"""
function extract_info(tree, snode::Int; counts=false, qvalues=false, safety=false, full_qvalues=false, full_safety=false, kwargs...)
    m = length(tree.children[snode])
    D = Dict()

    counts && (D[:N] = Vector{Float64}(undef, m))
    qvalues && (D[:Q] = Vector{Float64}(undef, m))
    full_qvalues && (D[:Qtree] = tree.q)
    full_safety && (D[:Ftree] = tree.f)
    safety && (D[:F] = Vector{Float64}(undef, m))
    safety && (D[:Δ] = Vector{Float64}(undef, m))
    safety && (D[:Δ0] = tree.Δ0)

    for (i,child) in enumerate(tree.children[snode])
        counts && (D[:N][i] = tree.n[child]) # N(b,a)
        qvalues && (D[:Q][i] = tree.q[child]) # Q(b,a)
        safety && (D[:F][i] = tree.f[child]) # F(b,a)
        safety && (D[:Δ][i] = tree.Δ[snode]) # Δ(b)
    end

    return NamedTuple{(collect(keys(D))...,)}((collect(values(D))...,))
end


"""
Given a criteria, extract Q, N, of S info based on `kwargs` and then call the dispacted `probability_vector` with the NamedTuple.
"""
probability_vector(crit::Criteria, tree, snode; kwargs...) = probability_vector(crit, extract_info(tree, snode; kwargs...))


"""
General `select_best` via maximum for a given vector `P`.
"""
function select_best(P::Vector, tree, snode)
    best = -Inf
    sanode = 0
    for (i,child) in enumerate(tree.children[snode])
        if P[i] > best
            best = P[i]
            sanode = child
        end
    end
    return sanode
end


"""
When setting temparature τ=0, we want to effectively select the max. So we use the `maxcrit` version of the criteria.
"""
function select_best_with_temp(crit::Criteria, maxcrit::Criteria, tree, snode::Int; kwargs...)
    if crit.τ == 0
        return select_best(maxcrit, tree, snode)
    else
        P = probability_vector(crit, tree, snode; kwargs...)
        return sample_best(P, tree, snode)
    end
end


"""
General `sample_best` for a given probability vector `P`.
"""
function sample_best(P::Vector, tree, snode)
    A = tree.children[snode]
    return rand(SparseCat(A, P))
end



###########################################################
###########################################################
###########################################################



"""
Return the best action based on the maximum Q-value.
"""
struct MaxQ <: Criteria end
probability_vector(crit::MaxQ, info::NamedTuple) = normalize(info.Q, 1)
select_best(crit::MaxQ, tree, snode::Int) = select_best(probability_vector(crit, tree, snode; qvalues=true), tree, snode)



###########################################################
###########################################################
###########################################################



"""
Return the best action based on the maximum visit count.
"""
struct MaxN <: Criteria end
select_best(crit::MaxN, tree, snode::Int) = select_best(probability_vector(crit, tree, snode; counts=true), tree, snode)
probability_vector(crit::MaxN, info::NamedTuple) = probability_vector(SampleN(τ=1), info)


"""
Return the best action based on exponentiated visit counts.
"""
@with_kw struct SampleN <: Criteria
    τ = 1
end
select_best(crit::SampleN, tree, snode::Int) = select_best_with_temp(crit, MaxN(), tree, snode; counts=true)
function probability_vector(crit::SampleN, info::NamedTuple)
    N = info.N
    τ = crit.τ
    P = N.^(1/τ) ./ sum(N.^(1/τ)) # exponentiated visit counts
    P = normalize(P, 1)
    return P
end



###########################################################
###########################################################
###########################################################



"""
Return the best action using information from both Q-values and visit counts.
"""
struct MaxQN <: Criteria end
select_best(crit::MaxQN, tree, snode::Int) = select_best(probability_vector(crit, tree, snode; qvalues=true, counts=true), tree, snode)
probability_vector(crit::MaxQN, info::NamedTuple) = probability_vector(SampleQN(τ=1), info)


@with_kw struct SampleQN <: Criteria
    τ = 1
end
select_best(crit::SampleQN, tree, snode::Int) = select_best_with_temp(crit, MaxQN(), tree, snode; qvalues=true, counts=true)
function probability_vector(crit::SampleQN, info::NamedTuple)
    Q, N = info.Q, info.N
    τ = crit.τ
    QN = (softmax(Q) .* (N ./ sum(N))) .^ (1/τ)
    P = normalize(QN, 1)
    return P
end



###########################################################
###########################################################
###########################################################



"""
Return the best action using information from both Q-values and visit counts.
"""
@with_kw struct MaxWeightedQN <: Criteria
    w = 0.5
end
select_best(crit::MaxWeightedQN, tree, snode::Int) = select_best(probability_vector(crit, tree, snode; qvalues=true, counts=true), tree, snode)
probability_vector(crit::MaxWeightedQN, info::NamedTuple) = probability_vector(SampleWeightedQN(τ=1, w=crit.w), info)


@with_kw struct SampleWeightedQN <: Criteria
    τ = 1
    w = 0.5
end
select_best(crit::SampleWeightedQN, tree, snode::Int) = select_best_with_temp(crit, MaxWeightedQN(w=crit.w), tree, snode; qvalues=true, counts=true)
function probability_vector(crit::SampleWeightedQN, info::NamedTuple)
    Q, N = info.Q, info.N
    τ = crit.τ
    w = crit.w
    QN = (w*softmax(Q) .+ (1-w)*(N ./ sum(N))) .^ (1/τ)
    P = normalize(QN, 1)
    return P
end



###########################################################
###########################################################
###########################################################


"""
Return the best action using information from both Q-values and visit counts, exponentiated by zq and zn.
"""
@with_kw struct MaxZQN <: Criteria
    zq = 1
    zn = 1
end
select_best(crit::MaxZQN, tree, snode::Int) = select_best(probability_vector(crit, tree, snode; qvalues=true, counts=true), tree, snode)
probability_vector(crit::MaxZQN, info::NamedTuple) = probability_vector(SampleZQN(τ=1, zq=crit.zq, zn=crit.zn), info)


@with_kw mutable struct SampleZQN <: Criteria
    τ = 1
    zq = 1
    zn = 1
end
select_best(crit::SampleZQN, tree, snode::Int) = select_best_with_temp(crit, MaxZQN(zq=crit.zq, zn=crit.zn), tree, snode; qvalues=true, counts=true)
function probability_vector(crit::SampleZQN, info::NamedTuple)
    Q, N = info.Q, info.N
    τ = crit.τ
    zq = crit.zq
    zn = crit.zn
    Nnorm = N ./ sum(N)
    QN = (softmax(Q).^zq .* Nnorm.^zn) .^ (1/τ)
    if all(QN .== 0) || all(isnan.(QN))
        QN = ones(length(QN))
    end
    P = normalize(QN, 1)
    return P
end



###########################################################
###########################################################
###########################################################


"""
Return the best action using information from Q-values, visit counts and safety. Exponentiated by zq and zn.
"""
@with_kw struct MaxZQNS <: Criteria
    zq = 1
    zn = 1
end
select_best(crit::MaxZQNS, tree, snode::Int) = select_best(probability_vector(crit, tree, snode; qvalues=true, counts=true, safety=true, full_qvalues=true, full_safety=true), tree, snode)
probability_vector(::MaxZQNS, info::NamedTuple) = probability_vector(SampleZQNS(τ=1), info)

sigmoid(z::Vector) = 1.0 ./ (1.0 .+ exp.(-z))
prob_normalize(X::Vector) = X ./ sum(X)
normalizeAB(X, a, b) = (X .- minimum(X)) / (maximum(X) - minimum(X)) * (b - a) .+ a

@with_kw mutable struct SampleZQNS <: Criteria
    τ = 1
    zq = 1
    zn = 1
end
select_best(crit::SampleZQNS, tree, snode::Int) = select_best_with_temp(crit, MaxZQNS(zq=crit.zq, zn=crit.zn), tree, snode; qvalues=true, counts=true, safety=true, full_qvalues=true, full_safety=true)
function probability_vector(crit::SampleZQNS, info)
    Q, N, F, Δ, Δ0 = info.Q, info.N, info.F, info.Δ, info.Δ0
    Qtree, Ftree = info.Qtree, info.Ftree
    τ = crit.τ
    zq = crit.zq
    zn = crit.zn
    Δ′ = max.(Δ0, Δ)

    # no safe actions
    if all(F .> Δ′)
        subject_to = ones(length(F))
    else
        subject_to = 𝟙.(F .≤ Δ′)
    end

    Qnorm = softmax(Q)
    Nnorm = prob_normalize(N)
    SQN = subject_to .* (Qnorm.^zq .* Nnorm.^zn).^(1/τ)

    if all(SQN .== 0) || all(isnan.(SQN))
        SQN = ones(length(SQN))
    end

    P = normalize(SQN, 1)
    return P
end
