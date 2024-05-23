𝟙(b) = b ? 1.0 : 0.0

POMDPs.solve(solver::CBZSolver, mdp::Union{POMDP,MDP}) = CBZPlanner(solver, mdp)

"""
Delete existing decision tree.
"""
function clear_tree!(p::CBZPlanner)
    p.tree = nothing
end

"""
Construct an MCTSCBZ tree and choose the best action.
"""
POMDPs.action(p::CBZPlanner, s) = first(action_info(p, s))

"""
Construct an MCTSCBZ tree and choose the best action. Also output some information.
"""
function POMDPTools.action_info(p::CBZPlanner, s; tree_in_info=false, counts_in_info=false)
    local a::actiontype(p.mdp)
    info = Dict{Symbol, Any}()
    try
        if isterminal(p.mdp, s)
            error("""
                  MCTS cannot handle terminal states. action was called with
                  s = $s
                  μ = $(s.ukf.μ)
                  Σ = $(s.ukf.Σ)
                  """)
        end

        S = statetype(p.mdp)
        A = actiontype(p.mdp)
        Δ0 = p.solver.Δ0
        if p.solver.keep_tree && !isnothing(p.tree)
            tree = p.tree
            if haskey(tree.s_lookup, s)
                snode = tree.s_lookup[s]
            else
                snode = insert_state_node!(tree, s, Δ0, true)
            end
        else
            tree = CBZTree{S,A}(p.solver.n_iterations; Δ0)
            p.tree = tree
            snode = insert_state_node!(tree, s, Δ0, p.solver.check_repeat_state)
        end

        timer = p.solver.timer
        p.solver.show_progress ? progress = Progress(p.solver.n_iterations) : nothing
        nquery = 0
        start_s = timer()
        for i = 1:p.solver.n_iterations
            nquery += 1
            simulate(p, snode, p.solver.depth) # (not 100% sure we need to make a copy of the state here)
            p.solver.show_progress ? next!(progress) : nothing
            if timer() - start_s >= p.solver.max_time
                p.solver.show_progress ? finish!(progress) : nothing
                break
            end
        end

        p.reset_callback(p.mdp, s) # Optional: leave the MDP in the current state.
        info[:search_time] = timer() - start_s
        info[:search_time_us] = info[:search_time]*1e6
        info[:tree_queries] = nquery
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end
        if p.solver.counts_in_info || counts_in_info
            root_children_idx = first(tree.children)
            root_actions = tree.a_labels[root_children_idx]
            root_counts = tree.n[root_children_idx]
            root_values = tree.q[root_children_idx]
            root_pfail = tree.f[root_children_idx]
            root_delta = tree.Δ[snode]
            info[:counts] = Dict(map(i->Pair(root_actions[i], (root_counts[i], root_values[i], root_pfail[i], root_delta, Δ0)), eachindex(root_actions)))
            info[:qvalues] = tree.q
            info[:fvalues] = tree.f
        end

        sanode = select_best(p.solver.final_criterion, tree, snode)
        a = tree.a_labels[sanode] # choose action with highest approximate value
    catch ex
        a = convert(actiontype(p.mdp), default_action(p.solver.default_action, p.mdp, s, ex))
        info[:exception] = ex
    end

    return a, info
end


"""
Return the reward for one iteration of MCTSCBZ.
"""
function simulate(cbz::CBZPlanner, snode::Int, d::Int)
    S = statetype(cbz.mdp)
    A = actiontype(cbz.mdp)
    sol = cbz.solver
    tree = cbz.tree
    s = tree.s_labels[snode]
    cbz.reset_callback(cbz.mdp, s) # Optional: used to reset/reinitialize MDP to a given state.
    if isterminal(cbz.mdp, s)
        return 0.0, 0.0
    elseif d == 0
        v = estimate_value(cbz.value_estimate, cbz.mdp, s, d)
        p = estimate_failure(cbz.failure_estimate, cbz.mdp, s)
        return v, p
    end

    a, sanode = action_selection(cbz, tree, snode, s)
    q, p = state_widen!(cbz, tree, sol, sanode, s, a, d)

    tree.total_n[snode] += 1
    tree.n[sanode] += 1
    tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]
    tree.f[sanode] += (p - tree.f[sanode])/tree.n[sanode]

    adaptation!(cbz, tree, snode, sanode)

    return q, p
end


"""
Adaptation of target failure probability threshold.
"""
function adaptation!(cbz::CBZPlanner, tree, snode::Int, sanode::Int)
    updatebounds!(tree, snode)
    Δ0, η = cbz.solver.Δ0, cbz.solver.η
    lb = tree.flb[snode]
    ub = tree.fub[snode]
    err = 𝟙(tree.f[sanode] > tree.Δ[snode])
    tree.Δ[snode] = clamp(tree.Δ[snode] + η*(err - Δ0), lb, ub)
    return tree.Δ[snode]
end


"""
Update failure probability bounds for adaptation.
"""
function updatebounds!(tree::CBZTree{S,A}, snode::Int) where {S,A}
    fb = [tree.f[sanode] for sanode in tree.children[snode]]
    tree.flb[snode] = minimum(fb)
    tree.fub[snode] = maximum(fb)
    return tree.flb[snode], tree.fub[snode]
end


"""
Action selection using safety-based PUCT.
"""
function action_selection(cbz::CBZPlanner, tree::CBZTree, snode::Int, s)
    # action progressive widening
    if cbz.solver.enable_action_pw
        action_widen!(cbz, tree, snode, s)
    elseif isempty(tree.children[snode])
        for a in actions(cbz.mdp, s)
            add_action!(cbz, tree, snode, s, a; check_repeat_action=false)
        end
    end
    sanode = best_sanode_SPUCT(cbz, tree, snode, s)
    a = tree.a_labels[sanode]
    return a, sanode
end


"""
Action progressive widening.
"""
function action_widen!(cbz::CBZPlanner, tree, snode::Int, s)
    sol = cbz.solver
    if length(tree.children[snode]) <= sol.k_action*tree.total_n[snode]^sol.alpha_action # criterion for new action generation
        a = next_action(cbz.next_action, cbz.mdp, s, CBZStateNode(tree, snode)) # action generation step
        add_action!(cbz, tree, snode, s, a)
    end
end


function add_action!(cbz::CBZPlanner, tree, snode::Int, s, a; check_repeat_action=cbz.solver.check_repeat_action)
    sol = cbz.solver
    if check_repeat_action && haskey(tree.a_lookup, (snode, a))
        sanode = tree.a_lookup[(snode, a)]
    else
        n0 = init_N(sol.init_N, cbz.mdp, s, a)
        q0 = init_Q(sol.init_Q, cbz.mdp, s, a)
        f0 = init_F(sol.init_F, cbz.mdp, s, a)
        sanode = insert_action_node!(tree, snode, a, n0, q0, f0, check_repeat_action)
        tree.total_n[snode] += n0
    end
    adaptation!(cbz, tree, snode, sanode)
end


"""
State progressive widening.
"""
function state_widen!(cbz::CBZPlanner, tree, sol, sanode, s, a, d)
    new_node = false
    if (sol.enable_state_pw && tree.n_a_children[sanode] <= sol.k_state*tree.n[sanode]^sol.alpha_state) || tree.n_a_children[sanode] == 0
        # sp, r, p = @gen(:sp, :r, :p)(cbz.mdp, s, a, cbz.rng) # TODO.
        sp, r, p = gen(cbz.mdp, s, a, cbz.rng)

        if sol.check_repeat_state && haskey(tree.s_lookup, sp)
            spnode = tree.s_lookup[sp]
        else
            spnode = insert_state_node!(tree, sp, sol.Δ0, sol.keep_tree || sol.check_repeat_state)
            new_node = true
        end

        push!(tree.transitions[sanode], (spnode, r, p))

        if !sol.check_repeat_state
            tree.n_a_children[sanode] += 1
        elseif !((sanode,spnode) in tree.unique_transitions)
            push!(tree.unique_transitions, (sanode,spnode))
            tree.n_a_children[sanode] += 1
        end
    else
        spnode, r, p = rand(cbz.rng, tree.transitions[sanode])
    end

    if new_node # if b ∉ 𝒯
        v′ = estimate_value(cbz.value_estimate, cbz.mdp, sp, d-1)
        p′ = estimate_failure(cbz.failure_estimate, cbz.mdp, sp)
    else
        v′, p′ = simulate(cbz, spnode, d-1)
    end

    γ = discount(cbz.mdp)
    q = r + γ*v′

    δ = sol.δ
    p = p + δ*(1-p)*p′

    return q, p
end


"""
Return the best action node based on the SPUCT score with exploration constant c
"""
function best_sanode_SPUCT(cbz::CBZPlanner, tree::CBZTree, snode::Int, s)
    mdp, sol = cbz.mdp, cbz.solver
    c = sol.exploration_constant
    Δ0 = tree.Δ0
    Δ′ = max(Δ0, tree.Δ[snode])
    𝐩 = estimate_policy(cbz.policy_estimate, mdp, s)

    best_SPUCT = -Inf
    sanode = 0

    Ns = sum(tree.total_n[snode])
    children = tree.children[snode]
    no_safe_actions = all(tree.f[child] > Δ′ for child in children)

    for child in children
        n = tree.n[child]
        q = tree.q[child]
        f = tree.f[child]
        if (Ns <= 0 && n == 0) || c == 0.0
            SPUCT = q
        else
            q = normalize01(q, tree.q; checkisnan=true)
            a = tree.a_labels[child]
            ai = findfirst(map(ab->a == ab, actions(mdp)))
            pa = 𝐩[ai]

            if no_safe_actions
                subject_to = 1
            else
                subject_to = 𝟙(f ≤ Δ′)
            end

            SPUCT = subject_to * (q + c*pa*sqrt(Ns)/(n+1))
        end
        @assert !isnan(SPUCT) "SPUCT was NaN (q=$q, pa=$pa, c=$c, Ns=$Ns, n=$n, sba=$sba)"
        @assert !isequal(SPUCT, -Inf)
        if SPUCT > best_SPUCT
            best_SPUCT = SPUCT
            sanode = child
        end
    end
    return sanode
end
