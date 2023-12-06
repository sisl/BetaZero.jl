POMDPs.solve(solver::CPUCTSolver, mdp::Union{POMDP,MDP}) = CPUCTPlanner(solver, mdp)

"""
Delete existing decision tree.
"""
function clear_tree!(p::CPUCTPlanner)
    p.tree = nothing
end

"""
Construct an MCTS-CPUCT tree and choose the best action.
"""
POMDPs.action(p::CPUCTPlanner, s) = first(action_info(p, s))

"""
Construct an MCTS-CPUCT tree and choose the best action. Also output some information.
"""
function POMDPTools.action_info(p::CPUCTPlanner, s; tree_in_info=false, counts_in_info=false)
    local a::actiontype(p.mdp)
    info = Dict{Symbol, Any}()
    try
        if isterminal(p.mdp, s)
            error("""
                  MCTS cannot handle terminal states. action was called with
                  s = $s
                  """)
        end

        S = statetype(p.mdp)
        A = actiontype(p.mdp)
        if p.solver.keep_tree && p.tree != nothing
            tree = p.tree
            if haskey(tree.s_lookup, s)
                snode = tree.s_lookup[s]
            else
                snode = insert_state_node!(tree, s, true)
            end
        else
            tree = CPUCTTree{S,A}(p.solver.n_iterations)
            p.tree = tree
            snode = insert_state_node!(tree, s, p.solver.check_repeat_state)
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
            root_idx = first(tree.children)
            root_actions = tree.a_labels[root_idx]
            root_counts = tree.n[root_idx]
            root_values = tree.q[root_idx]
            # root_pfail = tree.f[root_idx]
            root_pfail = exp.(tree.f[root_idx]) # Note exp done here
            # @show root_pfail
            info[:counts] = Dict(map(i->Pair(root_actions[i], (root_counts[i], root_values[i], root_pfail[i])), eachindex(root_actions)))
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
Return the reward for one iteration of MCTS-CPUCT.
"""
function simulate(dpw::CPUCTPlanner, snode::Int, d::Int)
    S = statetype(dpw.mdp)
    A = actiontype(dpw.mdp)
    sol = dpw.solver
    tree = dpw.tree
    s = tree.s_labels[snode]
    dpw.reset_callback(dpw.mdp, s) # Optional: used to reset/reinitialize MDP to a given state.
    if isterminal(dpw.mdp, s)
        return 0.0, log(eps())
        # return 0.0, 0.0
    elseif d == 0
        # return estimate_value(dpw.value_estimate, dpw.mdp, s, d), estimate_failure(dpw.failure_estimate, dpw.mdp, s)
        return estimate_value(dpw.value_estimate, dpw.mdp, s, d), log(estimate_failure(dpw.failure_estimate, dpw.mdp, s))
    end

    # action progressive widening
    if dpw.solver.enable_action_pw
        action_widen!(dpw, tree, snode, s)
    elseif isempty(tree.children[snode])
        for a in actions(dpw.mdp, s)
            n0 = init_N(sol.init_N, dpw.mdp, s, a)
            insert_action_node!(tree, snode, a, n0,
                                init_Q(sol.init_Q, dpw.mdp, s, a),
                                init_F(sol.init_F, dpw.mdp, s, a),
                                false)
            tree.total_n[snode] += n0
        end
    end
    
    p = estimate_policy(dpw.policy_estimate, dpw.mdp, s)
    # pfail = estimate_failure(dpw.failure_estimate, dpw.mdp, s)
    sanode = best_sanode_CPUCT(dpw.mdp, tree, snode, sol.exploration_constant, p)
    a = tree.a_labels[sanode]

    # q, pfail′ = state_widen!(dpw, tree, sol, sanode, s, a, d)
    q, logpfail′ = state_widen!(dpw, tree, sol, sanode, s, a, d)

    tree.n[sanode] += 1
    tree.total_n[snode] += 1
    tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]
    # tree.f[sanode] *= pfail′ # acculumate probability
    # tree.f[sanode] += logpfail′ # acculumate log-probability
    tree.f[sanode] += log(discount(dpw.mdp)) + logpfail′ # acculumate discounted log-probability
    # tree.f[sanode] = logpfail′ # DON'T ACCUMULATE

    # max-Q backups
    # maxq_node = select_best(MaxQ(), tree, snode)
    # return tree.q[maxq_node]

    return q, tree.f[sanode]
end



"""
Action progressive widening.
"""
function action_widen!(dpw::CPUCTPlanner, tree, snode::Int, s)
    sol = dpw.solver
    if length(tree.children[snode]) <= sol.k_action*tree.total_n[snode]^sol.alpha_action # criterion for new action generation
        a = next_action(dpw.next_action, dpw.mdp, s, CPUCTStateNode(tree, snode)) # action generation step
        add_action!(dpw, tree, snode, s, a)
    end
end


function add_action!(dpw::CPUCTPlanner, tree, snode::Int, s, a)
    sol = dpw.solver
    if !sol.check_repeat_action || !haskey(tree.a_lookup, (snode, a))
        n0 = init_N(sol.init_N, dpw.mdp, s, a)
        insert_action_node!(tree, snode, a, n0,
                            init_Q(sol.init_Q, dpw.mdp, s, a),
                            init_F(sol.init_F, dpw.mdp, s, a),
                            sol.check_repeat_action
                           )
        tree.total_n[snode] += n0
    end
end



"""
State progressive widening.
"""
function state_widen!(dpw::CPUCTPlanner, tree, sol, sanode, s, a, d)
    new_node = false
    if (sol.enable_state_pw && tree.n_a_children[sanode] <= sol.k_state*tree.n[sanode]^sol.alpha_state) || tree.n_a_children[sanode] == 0
    # if (sol.enable_state_pw && tree.n_a_children[sanode] <= sol.k_state) || tree.n_a_children[sanode] == 0 #! Note.
        sp, r = @gen(:sp, :r)(dpw.mdp, s, a, dpw.rng)

        if sol.check_repeat_state && haskey(tree.s_lookup, sp)
            spnode = tree.s_lookup[sp]
        else
            spnode = insert_state_node!(tree, sp, sol.keep_tree || sol.check_repeat_state)
            new_node = true
        end

        push!(tree.transitions[sanode], (spnode, r))

        if !sol.check_repeat_state
            tree.n_a_children[sanode] += 1
        elseif !((sanode,spnode) in tree.unique_transitions)
            push!(tree.unique_transitions, (sanode,spnode))
            tree.n_a_children[sanode] += 1
        end
    else
        spnode, r = rand(dpw.rng, tree.transitions[sanode])
    end

    if new_node
        v′ = estimate_value(dpw.value_estimate, dpw.mdp, sp, d-1)
        # pfail′ = estimate_failure(dpw.failure_estimate, dpw.mdp, sp)
        logpfail′ = log(estimate_failure(dpw.failure_estimate, dpw.mdp, sp))
    else
        # v′, pfail′ = simulate(dpw, spnode, d-1)
        v′, logpfail′ = simulate(dpw, spnode, d-1)
    end
    q = r + discount(dpw.mdp)*v′

    # return q, pfail′
    return q, logpfail′
end



function add_state!(sol::CPUCTSolver, tree, sanode, sp, r)
    new_node = false
    if sol.check_repeat_state && haskey(tree.s_lookup, sp)
        spnode = tree.s_lookup[sp]
    else
        spnode = insert_state_node!(tree, sp, sol.keep_tree || sol.check_repeat_state)
        new_node = true
    end

    push!(tree.transitions[sanode], (spnode, r))

    if !sol.check_repeat_state
        tree.n_a_children[sanode] += 1
    elseif !((sanode,spnode) in tree.unique_transitions)
        push!(tree.unique_transitions, (sanode,spnode))
        tree.n_a_children[sanode] += 1
    end

    return new_node, spnode
end



"""
Return the best action node based on the CPUCT score with exploration constant c
"""
function best_sanode_CPUCT(mdp::MDP, tree::CPUCTTree, snode::Int, c::Float64, p::Vector)
    best_CPUCT = -Inf
    sanode = 0
    Ns = sum(tree.total_n[snode])
    for child in tree.children[snode]
        n = tree.n[child]
        q = tree.q[child]
        f = tree.f[child]
        if (Ns <= 0 && n == 0) || c == 0.0
            CPUCT = q
        else
            q = normalize01(q, tree.q; checkisnan=true)
            # expf = normalize01(exp(f), exp.(tree.f); checkisnan=true)
            a = tree.a_labels[child]
            ai = findfirst(map(ab->a == ab, actions(mdp)))
            pa = p[ai]
            # CPUCT = q*(1 - f) + pa*c*sqrt(Ns)/(n+1)
            # @show (1 - exp(f))
            # @show exp(f)
            # @show f
            CPUCT = q*(1 - exp(f)) + pa*c*sqrt(Ns)/(n+1)
            # CPUCT = q*(1 - expf) + pa*c*sqrt(Ns)/(n+1)
        end
        @assert !isnan(CPUCT) "CPUCT was NaN (q=$q, pa=$pa, c=$c, Ns=$Ns, n=$n, f=$f)"
        @assert !isequal(CPUCT, -Inf) "CPUCT was -Inf (q=$q, pa=$pa, c=$c, Ns=$Ns, n=$n, f=$f)"
        if CPUCT > best_CPUCT
            best_CPUCT = CPUCT
            sanode = child
        end
    end
    return sanode
end
