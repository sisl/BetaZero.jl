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
        if p.solver.keep_tree && !isnothing(p.tree)
            tree = p.tree
            if haskey(tree.s_lookup, s)
                snode = tree.s_lookup[s]
            else
                snode = insert_state_node!(tree, s, p.solver.α0, true)
            end
        else
            tree = CBZTree{S,A}(p.solver.n_iterations)
            p.tree = tree
            snode = insert_state_node!(tree, s, p.solver.α0, p.solver.check_repeat_state)
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
            root_alpha = tree.α[snode]
            info[:counts] = Dict(map(i->Pair(root_actions[i], (root_counts[i], root_values[i], root_pfail[i], root_alpha)), eachindex(root_actions)))
        end

        # for sanode in tree.children[snode]
        #     adaptation!(p, tree, snode, sanode)
        # end

        sanode = select_best(p.solver.final_criterion, tree, snode)
        a = tree.a_labels[sanode] # choose action with highest approximate value

        if DEBUG_VERBOSE
            for sanode in tree.children[snode]
                a = tree.a_labels[sanode]
                @show a
                @show tree.f[sanode]
                # @show isfailure(p.mdp, s, a)
            end
            println("."^40)
        end
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
        e = estimate_failure(cbz.failure_estimate, cbz.mdp, s)
        return v, e # ! TODO: e to p_fail and ep to pp_fail
    end
    
    # tree.total_n[snode] += 1 # ! NOTE.

    a, sanode = action_selection(cbz, tree, snode, s)    
    q, ep = state_widen!(cbz, tree, sol, sanode, s, a, d)
    
    tree.total_n[snode] += 1 # ! NOTE.
    tree.n[sanode] += 1
    tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]
    tree.f[sanode] += (ep - tree.f[sanode])/tree.n[sanode]
    
    # updatebounds!(tree, snode, sanode)
    adaptation!(cbz, tree, snode, sanode)

    return q, ep
end


"""
Adaptation
"""
function adaptation!(cbz::CBZPlanner, tree, snode::Int, sanode::Int)
    updatebounds!(tree, snode, sanode)
    η, α0 = cbz.solver.η, cbz.solver.α0
    lb, ub = tree.flb[snode], tree.fub[snode]
    err = 𝟙(tree.f[sanode] > tree.α[snode])
    # @info "Before ($(err == 1)): $(tree.α[snode])"
    # tree.α[snode] = clamp(tree.α[snode] + η*(err + α0 - 1), lb, ub)
    # tree.α[snode] = clamp(tree.α[snode] + η*(err + α0 - 1), lb, 1)
    # @info "Before: $(tree.α[snode])"
    # @show lb, ub
    tree.α[snode] = clamp(tree.α[snode] + η*(err - α0), lb, ub)
    # tree.α[snode] = clamp(tree.α[snode] + η*(err - α0), lb, 1)
    # tree.α[snode] = tree.α[snode] + η*(err - α0)
    # @info "After: $(err == 1) and $(tree.α[snode])"
    # tree.α[snode] = tree.α[snode] + η*(err + α0 - 1)
    # @info "After ($(err == 1)): $(tree.α[snode])"
    return tree.α[snode]
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
        sanode = insert_action_node!(tree, snode, a, n0,
                                     init_Q(sol.init_Q, cbz.mdp, s, a),
                                     init_F(sol.init_F, cbz.mdp, s), # ! NOTE.
                                     check_repeat_action
                                    )
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
        # sp, r, e = @gen(:sp, :r, :e)(cbz.mdp, s, a, cbz.rng) # TODO.
        sp, r, _ = gen(cbz.mdp, s, a, cbz.rng)
        e = isfailure(cbz.mdp, s, a)

        if sol.check_repeat_state && haskey(tree.s_lookup, sp)
            spnode = tree.s_lookup[sp]
        else
            spnode = insert_state_node!(tree, sp, sol.α0, sol.keep_tree || sol.check_repeat_state)
            new_node = true
        end

        push!(tree.transitions[sanode], (spnode, r, e))

        if !sol.check_repeat_state
            tree.n_a_children[sanode] += 1
        elseif !((sanode,spnode) in tree.unique_transitions)
            push!(tree.unique_transitions, (sanode,spnode))
            tree.n_a_children[sanode] += 1
        end
    else
        spnode, r, e = rand(cbz.rng, tree.transitions[sanode])
    end

    if new_node # if b ∉ 𝒯
        vp = estimate_value(cbz.value_estimate, cbz.mdp, sp, d-1)
        ep = estimate_failure(cbz.failure_estimate, cbz.mdp, sp)
    else
        vp, ep = simulate(cbz, spnode, d-1)
    end
    γ = discount(cbz.mdp)
    q = r + γ*vp
    e = (e + ep) / 2

    return q, e
end


"""
Return the best action node based on the SPUCT score with exploration constant c
"""
function best_sanode_SPUCT(cbz::CBZPlanner, tree::CBZTree, snode::Int, s)
    mdp, sol = cbz.mdp, cbz.solver
    c = sol.exploration_constant
    p = estimate_policy(cbz.policy_estimate, mdp, s)

    num_equal_values = 1
    best_SPUCT = -Inf
    sanode = 0
    Ns = sum(tree.total_n[snode])
    no_safe_actions = all(tree.f[child] > tree.α[snode] for child in tree.children[snode])
    children = tree.children[snode]

    for child in children
        n = tree.n[child]
        q = tree.q[child]
        if (Ns <= 0 && n == 0) || c == 0.0
            SPUCT = q
        else
            q = normalize01(q, tree.q; checkisnan=true)
            a = tree.a_labels[child]
            ai = findfirst(map(ab->a == ab, actions(mdp)))
            pa = p[ai]
            if no_safe_actions
                # Sba = 1 # ! NOTE
                Sba = 1 - tree.f[child]
                # Sba = tree.α[snode] - tree.f[child]
            else
                # Sba = 𝟙(tree.f[child] ≤ tree.α[snode]) * (1 - tree.f[child]) # ! NOTE.
                # Sba = 𝟙(tree.f[child] ≤ tree.α[snode]) * (tree.α[snode] - tree.f[child]) # ! NOTE.

                Sba = 𝟙(tree.f[child] ≤ tree.α[snode]) * (1 - tree.f[child]) # ! NOTE.
                # Sba = 𝟙(tree.f[child] ≤ tree.α[snode])
            end
            # SPUCT = q + c*pa*sqrt(Ns)/(n+1) # ! NOTE.
            SPUCT = Sba*(q + c*pa*sqrt(Ns)/(n+1))
        end
        @assert !isnan(SPUCT) "SPUCT was NaN (q=$q, pa=$pa, c=$c, Ns=$Ns, n=$n, Sba=$Sba)"
        @assert !isequal(SPUCT, -Inf)
        if SPUCT ≈ best_SPUCT
            # Count identical values
            num_equal_values += 1
        elseif SPUCT > best_SPUCT
        # if SPUCT > best_SPUCT
            best_SPUCT = SPUCT
            sanode = child
        end
    end
    if length(children) > 1 && num_equal_values == length(children)
        # randomly pick action if their SPUCT are all equal
        sanode = rand(children)
    end
    return sanode
end
