POMDPs.solve(solver::DARSolver, mdp::Union{POMDP,MDP}) = DARPlanner(solver, mdp)

"""
Delete existing decision tree.
"""
function clear_tree!(p::DARPlanner)
    p.tree = nothing
end

"""
Construct an MCTSDAR tree and choose the best action.
"""
POMDPs.action(p::DARPlanner, s) = first(action_info(p, s))

"""
Construct an MCTSDAR tree and choose the best action. Also output some information.
"""
function POMDPTools.action_info(p::DARPlanner, s; tree_in_info=false, counts_in_info=false)
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
            tree = DARTree{S,A}(p.solver.n_iterations)
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
            info[:counts] = Dict(map(i->Pair(root_actions[i], (root_counts[i], root_values[i])), eachindex(root_actions)))
        end

        sanode = best_sanode(tree, snode)
        a = tree.a_labels[sanode] # choose action with highest approximate value
    catch ex
        a = convert(actiontype(p.mdp), default_action(p.solver.default_action, p.mdp, s, ex))
        info[:exception] = ex
    end

    return a, info
end


"""
Return the reward for one iteration of MCTSDAR.
"""
function simulate(dpw::DARPlanner, snode::Int, d::Int)
    S = statetype(dpw.mdp)
    A = actiontype(dpw.mdp)
    sol = dpw.solver
    tree = dpw.tree
    s = tree.s_labels[snode]
    dpw.reset_callback(dpw.mdp, s) # Optional: used to reset/reinitialize MDP to a given state.
    if isterminal(dpw.mdp, s)
        return 0.0
    elseif d == 0
        return estimate_value(dpw.value_estimate, dpw.mdp, s, d)
    end

    # action progressive widening
    if dpw.solver.enable_action_pw

        action_abstraction_refine!(dpw, tree, snode, s)
        # action_widen!(dpw, tree, snode, s)

    elseif isempty(tree.children[snode])
        for a in actions(dpw.mdp, s)
            n0 = init_N(sol.init_N, dpw.mdp, s, a)
            insert_action_node!(tree, snode, a, n0,
                                init_Q(sol.init_Q, dpw.mdp, s, a),
                                false)
            tree.total_n[snode] += n0
        end
    end
    
    sanode = best_sanode_UCB(dpw.mdp, tree, snode, sol.exploration_constant)
    a = tree.a_labels[sanode]

    # q = state_widen!(dpw, tree, sol, sanode, s, a, d)
    q = state_abstraction_refine!(dpw, tree, sol, sanode, s, a, d)

    tree.n[sanode] += 1
    tree.total_n[snode] += 1
    tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]

    return q
end



dist_manhattan(v, v′) = norm(v - v′, 1)
dist_euclidean(v, v′) = norm(v - v′, 2)
# dist_euclidean_praticles(v, v′) = norm(map(p->p.y, v.particles.particles) - map(p->p.y, v′.particles.particles), 2) # TODO.
dist_euclidean_praticles(v, v′) = norm(mean(v).y - mean(v′).y, 2) # TODO.
dist_supremum(v, v′)  = norm(v - v′, Inf)

# function action_samples(a, a′)
#     _v = POMDPs.convert_a(Vector, a)
#     _v′ = POMDPs.convert_a(Vector, a′)
#     value_indices = 1:3:length(_v) # only use the value of the sample, not LL.
#     v = _v[value_indices]
#     v′ = _v′[value_indices]
#     return v, v′
# end

# dist_manhattan(a, a′) = dist_manhattan(action_samples(a, a′)...)
# dist_euclidean(a, a′) = dist_euclidean(action_samples(a, a′)...)
# dist_supremum(a, a′) = dist_supremum(action_samples(a, a′)...)

k_nearest_neighbors(x, D, dist, k) = D[partialsortperm([dist(x, x′) for x′ in D], 1:k)]
nearest_neighbor(x, D, dist) = D[argmin([dist(x, x′) for x′ in D])]
ϵdecay_action(n) = n^-0.1 # from [Sokota et al.]
ϵdecay_state(n) = n^-10 # from [Sokota et al.]
# ϵdecay(n; λ=1/10_000) = exp(-λ*n)


"""
Action abstraction refining adapted from Sokota et al.
"""
function action_abstraction_refine!(dpw::DARPlanner, tree, snode::Int, s)
    sol = dpw.solver
    a = next_action(dpw.next_action, dpw.mdp, s, DARStateNode(tree, snode))
    s_idx = tree.s_lookup[s]
    children_idx = tree.children[s_idx]
    children = tree.a_labels[children_idx] # children of s
    if isempty(children)
        add_action!(dpw, tree, snode, s, a) # add a to children of s
        return a # return new action
    else
        useknn = false
        if useknn
            k = min(length(children), 2)
            knns = k_nearest_neighbors(a, children, dist_euclidean, k)
            ℓ = [sum(knn.sample[k].logprob[1] for k in keys(knn.sample)) for knn in knns]
            ℓᵢ = rand(Categorical(softmax(ℓ)))
            a′ = knns[ℓᵢ]
        else
            a′ = nearest_neighbor(a, children, dist_euclidean) # nearest child action
        end
        a′_idx = tree.a_lookup[(s_idx, a′)]
        n = tree.n[a′_idx] # N(s,a′)
        if dist_euclidean(a, a′) < ϵdecay_action(n)
            return a′ # return nearest
        else
            add_action!(dpw, tree, snode, s, a) # add a to children of s
            return a # return new action    
        end
    end
end


"""
Action progressive widening.
"""
function action_widen!(dpw::DARPlanner, tree, snode::Int, s)
    sol = dpw.solver
    if length(tree.children[snode]) <= sol.k_action*tree.total_n[snode]^sol.alpha_action # criterion for new action generation
        a = next_action(dpw.next_action, dpw.mdp, s, DARStateNode(tree, snode)) # action generation step
        add_action!(dpw, tree, snode, s, a)
    end
end


function add_action!(dpw::DARPlanner, tree, snode::Int, s, a)
    sol = dpw.solver
    if !sol.check_repeat_action || !haskey(tree.a_lookup, (snode, a))
        n0 = init_N(sol.init_N, dpw.mdp, s, a)
        insert_action_node!(tree, snode, a, n0,
                            init_Q(sol.init_Q, dpw.mdp, s, a),
                            sol.check_repeat_action
                           )
        tree.total_n[snode] += n0
    end
end




"""
State abstraction refinement from Sokota et al.
"""
function state_abstraction_refine!(dpw::DARPlanner, tree, sol, sanode, s, a, d)
    new_node = false
    sp, r = @gen(:sp, :r)(dpw.mdp, s, a, dpw.rng)
    children_idx = first.(tree.transitions[sanode])
    children = tree.s_labels[children_idx] # children of (s,a)
    if isempty(children)
        new_node, spnode = add_state!(sol, tree, sanode, sp, r)
    else
        nearest_neighbor_idx(x, D, dist) = argmin([dist(x, x′) for x′ in D])
        spp_idx = nearest_neighbor_idx(sp, children, dist_euclidean_praticles) # nearest child state
        spp = children[spp_idx]
        n = tree.n_a_children[sanode] # N(s,a,spp)
        if dist_euclidean_praticles(sp, spp) < ϵdecay_state(n)
            spnode, r = tree.transitions[sanode][spp_idx]
            # spnode, r = rand(dpw.rng, tree.transitions[sanode])
            # for (spnode, r) in tree.transitions[sanode]
                # if tree.s_labels[spnode] == spp
                    # break
                # end
            # end
            # sppnode = tree.s_lookup[spp]
            # spnode, r = tree.transitions[sanode]
        else
            new_node, spnode = add_state!(sol, tree, sanode, sp, r) # add sp to children of (s,a)
        end
    end

    if new_node
        q = r + discount(dpw.mdp)*estimate_value(dpw.value_estimate, dpw.mdp, sp, d-1)
    else
        q = r + discount(dpw.mdp)*simulate(dpw, spnode, d-1)
    end

    return q
end


"""
State progressive widening.
"""
function state_widen!(dpw, tree, sol, sanode, s, a, d)
    new_node = false
    if (dpw.solver.enable_state_pw && tree.n_a_children[sanode] <= sol.k_state*tree.n[sanode]^sol.alpha_state) || tree.n_a_children[sanode] == 0
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
        q = r + discount(dpw.mdp)*estimate_value(dpw.value_estimate, dpw.mdp, sp, d-1)
    else
        q = r + discount(dpw.mdp)*simulate(dpw, spnode, d-1)
    end

    return q
end



function add_state!(sol, tree, sanode, sp, r)
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
Return the best action.

Some publications say to choose action that has been visited the most
e.g., Continuous Upper Confidence Trees by Couëtoux et al.
"""
function best_sanode(tree::DARTree, snode::Int)
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


"""
Return the best action node based on the UCB score with exploration constant c
"""
function best_sanode_UCB(mdp::MDP, tree::DARTree, snode::Int, c::Float64)
    best_UCB = -Inf
    sanode = 0
    ltn = log(tree.total_n[snode])
    for child in tree.children[snode]
        n = tree.n[child]
        q = tree.q[child]
        if (ltn <= 0 && n == 0) || c == 0.0
            UCB = q
        else
            UCB = q + c*sqrt(ltn/n)
        end
        @assert !isnan(UCB) "UCB was NaN (q=$q, c=$c, ltn=$ltn, n=$n)"
        @assert !isequal(UCB, -Inf)
        if UCB > best_UCB
            best_UCB = UCB
            sanode = child
        end
    end
    return sanode
end
