"""
Use the raw policy head of the network to get the next action given a belief.
"""
mutable struct RawNetworkPolicy <: Policy
    pomdp::POMDP # TODO: Union{MDP,POMDP}
    surrogate::Surrogate
end


function POMDPs.action(policy::RawNetworkPolicy, b)
    problem = policy.pomdp
    A = POMDPs.actions(problem)
    Ab = POMDPs.actions(problem, b)
    p = policy_lookup(policy.surrogate, b)

    # Match indices of (potentially) reduced belief-dependent action space to get correctly associated probabilities from the network
    if length(A) != length(Ab)
        idx = Vector{Int}(undef, length(Ab))
        for (i,a) in enumerate(A)
            for (j,ab) in enumerate(Ab)
                if a == ab
                    idx[j] = i
                    break
                end
            end
        end
        p = p[idx]
    end

    # pidx = sortperm(p)
    # # BetaZero.UnicodePlots.barplot(actions(pomdp)[pidx], _P[pidx]) |> display
    # BetaZero.UnicodePlots.barplot(Ab[pidx], p[pidx]) |> display

    exponentiate_policy = false
    if exponentiate_policy
        τ = 2
        p = normalize(p .^ τ, 1)
        return rand(SparseCat(Ab, p))
    else
        return Ab[argmax(p)]
    end
end



@with_kw mutable struct RawValueNetworkPolicy <: Policy
    mdp::MDP
    surrogate::Surrogate
    n_obs::Int = 1 # Number of observations per action to branch (equal to number of belief updates)
end

RawValueNetworkPolicy(mdp::MDP, surrogate::Surrogate) = RawValueNetworkPolicy(mdp=mdp, surrogate=surrogate)


function POMDPs.action(policy::RawValueNetworkPolicy, s; include_info::Bool=false, counts_in_info::Bool=true, tree_in_info::Bool=true, run_parallel::Bool=false)
    estimate_value::Function = sp->value_lookup(policy.surrogate, sp) # Leaf node value estimator
    mdp = policy.mdp
    rng = Random.GLOBAL_RNG # TODO; parameterize
    tree = Dict()
    counts = Dict()
    info = Dict()
    λ_lcb = 1.0

    for a in actions(mdp, s)
        tree[a] = (sp=[], q=[])
        values = []
        # TODO: threads instead
        if run_parallel
            results = pmap(_->begin
                sp, r = @gen(:sp, :r)(mdp, s, a, rng)
                q = r + discount(mdp)*estimate_value(sp)
                (sp, q)
            end, 1:policy.n_obs)

            for (sp,q) in results
                push!(tree[a].sp, sp)
                push!(tree[a].q, q)
                push!(values, q)
            end
        else
            for _ in 1:policy.n_obs
                sp, r = @gen(:sp, :r)(mdp, s, a, rng)
                q = r + discount(mdp)*estimate_value(sp)
                push!(tree[a].sp, sp)
                push!(tree[a].q, q)
                push!(values, q)
            end
        end

        count = policy.n_obs
        μ, σ = mean_and_std(values)
        counts[a] = (count, μ, μ - λ_lcb*σ)
    end

    # select action based on maximum average Q-value
    best_a = reduce((a,a′) -> mean(tree[a].q) ≥ mean(tree[a′].q) ? a : a′, keys(tree))

    if include_info
        if counts_in_info # TODO: p.solver.counts_in_info ||
            info[:counts] = counts
        end

        if tree_in_info # TODO: p.solver.tree_in_info ||
            info[:tree] = tree
        end

        return (best_a, info)
    else
        return best_a
    end
end


function policy_lookup(policy::RawValueNetworkPolicy, b)
    _, info = action_info(policy, b)
    A = actions(policy.mdp)
    Ab = collect(keys(info[:counts]))
    Qb = last.(collect(values(info[:counts])))
    Qb = softmax(Qb) # Note, softmax applied here so illegal actions can be zeroed out.
    Q = zeros(length(A))
    for (i,a) in enumerate(A)
        for (j,ab) in enumerate(Ab)
            if a == ab
                Q[i] = Qb[j]
                break
            end
        end
    end
    return Q
end


@with_kw mutable struct RawPfailNetworkPolicy <: Policy
    mdp::MDP
    surrogate::Surrogate
    n_obs::Int = 1 # Number of observations per action to branch (equal to number of belief updates)
    δ::Real = 0.5
end

RawPfailNetworkPolicy(mdp::MDP, surrogate::Surrogate) = RawPfailNetworkPolicy(mdp=mdp, surrogate=surrogate)


function POMDPs.action(policy::RawPfailNetworkPolicy, s; include_info::Bool=false, counts_in_info::Bool=true, tree_in_info::Bool=true, run_parallel::Bool=false)
    estimate_value = 0
    estimate_failure::Function = sp->pfail_lookup(policy.surrogate, sp) # Leaf node p(fail) estimator
    mdp = policy.mdp
    rng = Random.GLOBAL_RNG # TODO; parameterize
    tree = Dict()
    counts = Dict()
    info = Dict()
    λ_lcb = 1.0

    for a in actions(mdp, s)
        tree[a] = (sp=[], q=[], f=[])
        qvalues = []
        fvalues = []
        # TODO: threads instead
        if run_parallel
            results = pmap(_->begin
                sp, r = @gen(:sp, :r)(mdp, s, a, rng)
                p = MCTS.isfailure(mdp, s, a)
                q = r + discount(mdp)*estimate_value
                δ = policy.δ
                p′ = estimate_failure(sp)
                f = (1-δ)*p + δ*(1-p)*p′
                (sp, q, f)
            end, 1:policy.n_obs)

            for (sp,q,f) in results
                push!(tree[a].sp, sp)
                push!(tree[a].q, q)
                push!(tree[a].f, f)
                push!(qvalues, q)
                push!(fvalues, f)
            end
        else
            for _ in 1:policy.n_obs
                sp, r = @gen(:sp, :r)(mdp, s, a, rng)
                p = MCTS.isfailure(mdp, s, a)
                q = r + discount(mdp)*estimate_value
                δ = policy.δ
                p′ = estimate_failure(sp)
                f = (1-δ)*p + δ*(1-p)*p′
                push!(tree[a].sp, sp)
                push!(tree[a].q, q)
                push!(tree[a].f, f)
                push!(qvalues, q)
                push!(fvalues, f)
            end
        end

        count = policy.n_obs
        μ, σ = mean_and_std(fvalues)
        counts[a] = (count, μ, μ - λ_lcb*σ)
    end

    # select action based on maximum average Q-value
    best_a = reduce((a,a′) -> mean(tree[a].f) ≤ mean(tree[a′].f) ? a : a′, keys(tree))

    if include_info
        if counts_in_info # TODO: p.solver.counts_in_info ||
            info[:counts] = counts
        end

        if tree_in_info # TODO: p.solver.tree_in_info ||
            info[:tree] = tree
        end

        return (best_a, info)
    else
        return best_a
    end
end


POMDPTools.action_info(policy::Union{RawValueNetworkPolicy,RawPfailNetworkPolicy}, b) = action(policy, b; include_info=true)
