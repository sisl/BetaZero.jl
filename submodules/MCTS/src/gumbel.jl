POMDPs.solve(solver::GumbelSolver, mdp::Union{POMDP,MDP}) = GumbelPlanner(solver, mdp)

"""
Delete existing decision tree.
"""
function clear_tree!(p::GumbelPlanner)
    p.tree = nothing
end

"""
Construct an Gumbel-MCTS tree and choose the best action.
"""
POMDPs.action(p::GumbelPlanner, s) = first(action_info(p, s))


# TODO: parameterize
c_visit = 50
c_scale = 0.1 # 10.0 # 0.01
num_action_expansions = 4
num_state_expansions = 200
use_state_pw = true

function invert_softmax(p)
    ϵ = 1f-45
    c = (1 - sum(log.(max.(ϵ, p)))) / length(p)
    return log.(p) .+ c
end


"""
Construct an Gumbel-MCTS tree and choose the best action. Also output some information.
"""
function POMDPTools.action_info(p::GumbelPlanner, s; tree_in_info=false, counts_in_info=false)
    local mdp = p.mdp
    local a::actiontype(mdp)
    info = Dict{Symbol, Any}()
    try
        if isterminal(mdp, s)
            error("""
                  MCTS cannot handle terminal states. action was called with
                  s = $s
                  """)
        end

        S = statetype(mdp)
        A = actiontype(mdp)
        if p.solver.keep_tree && !isnothing(p.tree)
            tree = p.tree
            if haskey(tree.s_lookup, s)
                snode = tree.s_lookup[s]
            else
                snode = insert_state_node!(tree, s, true)
            end
        else
            tree = GumbelTree{S,A}(p.solver.n_iterations)
            p.tree = tree
            snode = insert_state_node!(tree, s, p.solver.check_repeat_state)
        end

        timer = p.solver.timer
        p.solver.show_progress ? progress = Progress(p.solver.n_iterations) : nothing
        nquery = 0
        start_s = timer()

        n = p.solver.n_iterations
        # TODO: parameterize
        s = tree.s_labels[snode]
        A = actions(mdp, s)
        valid = valid_action_indices(mdp, s)

        policy_vector = estimate_policy(p.policy_estimate, mdp, s)
        full_logits = invert_softmax(policy_vector) # TODO: Is this even necessary? argmax should still preserve the right order, correct?
        logits = full_logits[valid]
        
        k = length(A)
        G = Gumbel(0)
        g = rand(G, k)
        
        m = min(k, num_action_expansions)

        # TODO: Parameterize (default = 1/2)
        λ = 1/2 # fraction to remove when sequential halving

        # TODO: Function.
        q̂(a) = haskey(tree.a_lookup, (snode, a)) ? tree.q[tree.a_lookup[(snode, a)]] : estimate_value(p.value_estimate, mdp, s, 0)
        σ(a) = (c_visit + maximum(tree.n)) * c_scale*q̂(a) # TODO: σ(q)

        local top_action_idx
        local top_actions
        top_action_idx = partialsortperm([g[ai] + logits[ai] for ai in 1:k], 1:m, rev=true)
        top_actions = A[top_action_idx]

        local mi = m

        # phases = ceil(Int, log(2, m))
        phases = ceil(Int, log(1/(1 - λ), m))
        for i in 1:phases
            nquery += 1

            if i > 1
                # mi = ceil(Int, m/(2^(i-1))) # Take ceiling for odd splits. (TODO?)
                # mi = m÷(2^(i-1)) # assumes to remove half
                mi = floor(Int, m*((1-λ)^(i-1)))
                top_action_idx = partialsortperm([g[ai] + full_logits[ai] + σ(A[ai]) for ai in 1:k], 1:mi, rev=true)
                top_actions = A[top_action_idx]
            end

            # @show top_actions
            simulate(p, snode, top_actions, p.solver.depth; isroot=true, g, i, m, mi, phases, λ)

            p.solver.show_progress ? next!(progress) : nothing
            if timer() - start_s >= p.solver.max_time
                p.solver.show_progress ? finish!(progress) : nothing
                break
            end
        end

        # TODO: Handle extra
        extra = n - sum(begin
            mi = floor(Int, m*((1-λ)^(i-1)))
            mi * floor(Int, n / (ceil(log(1/(1 - λ),m)) * mi))
        end for i in 1:phases)
        # @show extra

        p.reset_callback(mdp, s) # Optional: leave the MDP in the current state.
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


        # Important: uses logits for _all_ actions
        full_A = actions(mdp)
        ϵ_logit = -1e9
        full_logits = full_logits .- maximum(full_logits)
        # TODO: remove ϵ_logit?
        π′ = softmax([max(ϵ_logit, full_logits[i] + σ(a)) for (i,a) in enumerate(full_A)])
        info[:completed_policy] = π′ # TODO: parameter.


        # TODO: Function.
        if length(top_actions) == 1
            a = top_actions[1]
        else
            logits = full_logits[top_action_idx]
            best_a = nothing
            best_q = -Inf

            for (ai, top_a) in enumerate(top_actions)
                est_q = g[ai] + logits[ai] + σ(top_a)
                if est_q > best_q
                    best_q = est_q
                    best_a = top_a
                end
            end
            a = best_a
        end

        # sanode = best_sanode(tree, snode)
        # a = tree.a_labels[sanode] # choose action with highest approximate value

        # @info a == a_via_q

    catch ex
        a = convert(actiontype(mdp), default_action(p.solver.default_action, mdp, s, ex))
        info[:exception] = ex
    end

    return a, info
end

"""
Return the reward for one iteration of Gumbel-MCTS.
"""
function simulate(planner::GumbelPlanner, snode::Int, top_actions::Vector, d::Int; isroot::Bool=false, g::Vector, i::Int, m::Int, mi::Int, phases::Int, λ::Real)
    mdp = planner.mdp
    sol = planner.solver
    tree = planner.tree
    s = tree.s_labels[snode]
    planner.reset_callback(mdp, s) # Optional: used to reset/reinitialize MDP to a given state.

    n = planner.solver.n_iterations
    # repeated = floor(Int, n / (log(2,m)*m/i)) # TODO.
    # repeated = floor(Int, n / (ceil(Int, log(1/(1 - λ),m))*m/i)) # TODO.
    repeated = floor(Int, n / (ceil(Int, log(1/(1 - λ),m))*mi)) # TODO.

    # action selection from sequential halving
    for a in top_actions
        if !sol.check_repeat_action || !haskey(tree.a_lookup, (snode, a))
            n0 = init_N(sol.init_N, mdp, s, a)
            insert_action_node!(tree, snode, a, n0,
                                init_Q(sol.init_Q, mdp, s, a),
                                sol.check_repeat_action
                                )
            tree.total_n[snode] += n0
        end

        sanode = tree.a_lookup[(snode, a)]

        # @show repeated
        for _ in 1:repeated # TODO: thread parallelize
            
            if use_state_pw
                # NOTE: Using state PW
                state_expandion_condition = (sol.enable_state_pw && tree.n_a_children[sanode] <= sol.k_state*tree.n[sanode]^sol.alpha_state) || tree.n_a_children[sanode] == 0
            else
                # TODO: div not necessary for top level simulate
                # j = max(1, num_state_expansions÷(2^(sol.depth - d)) ÷ (phases - i + 1))
                j = max(1, floor(Int, num_state_expansions^(1/(phases - i + 1)))) # TODO: simplify
                state_expandion_condition = (tree.n_a_children[sanode] < j) || tree.n_a_children[sanode] == 0
            end
            
            # state expansion
            new_node = false
            if state_expandion_condition
                sp, r = @gen(:sp, :r)(mdp, s, a, planner.rng)

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
                spnode, r = rand(planner.rng, tree.transitions[sanode])
            end

            if new_node
                q = r + discount(mdp)*estimate_value(planner.value_estimate, mdp, sp, d-1)
            else
                q = r + discount(mdp)*inner_simulate(planner, spnode, d-1; phases, i)
            end

            tree.n[sanode] += 1
            tree.total_n[snode] += 1
            tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]
        end
    end
end



function inner_simulate(planner::GumbelPlanner, snode::Int, d::Int; i::Int, phases::Int)
    mdp = planner.mdp
    sol = planner.solver
    tree = planner.tree
    s = tree.s_labels[snode]
    planner.reset_callback(mdp, s) # Optional: used to reset/reinitialize MDP to a given state.

    if isterminal(mdp, s)
        return 0.0
    elseif d == 0
        return estimate_value(planner.value_estimate, mdp, s, d)
    end

    # action selection
    a = select_action(planner, snode, d)
    if !sol.check_repeat_action || !haskey(tree.a_lookup, (snode, a))
        n0 = init_N(sol.init_N, mdp, s, a)
        insert_action_node!(tree, snode, a, n0,
                            init_Q(sol.init_Q, mdp, s, a),
                            sol.check_repeat_action
                            )
        tree.total_n[snode] += n0
    end
    sanode = tree.a_lookup[(snode, a)]

    # state expansion
    if use_state_pw
        # NOTE: Using state PW
        state_expandion_condition = (sol.enable_state_pw && tree.n_a_children[sanode] <= sol.k_state*tree.n[sanode]^sol.alpha_state) || tree.n_a_children[sanode] == 0
    else
        # j = max(1, num_state_expansions÷(2^(sol.depth - d)) ÷ (phases - i + 1))    # TODO: parameterize
        j = max(1, floor(Int, num_state_expansions^(1/(phases - i + 1)))) # TODO: simplify
        state_expandion_condition = (tree.n_a_children[sanode] < j) || tree.n_a_children[sanode] == 0
    end

    new_node = false
    if state_expandion_condition
        sp, r = @gen(:sp, :r)(mdp, s, a, planner.rng)

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
        spnode, r = rand(planner.rng, tree.transitions[sanode])
    end

    if new_node
        q = r + discount(mdp)*estimate_value(planner.value_estimate, mdp, sp, d-1)
    else
        q = r + discount(mdp)*inner_simulate(planner, spnode, d-1; i, phases)
    end

    tree.n[sanode] += 1
    tree.total_n[snode] += 1
    tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]

    return q
end


function select_action(planner::GumbelPlanner, snode::Int, d::Int)
    mdp = planner.mdp
    sol = planner.solver
    tree = planner.tree
    s = tree.s_labels[snode]

    # TODO: move out as functions/cleanup
    normalize_q(q) = normalize01(q, tree.q; checkisnan=true)
    N(a) = haskey(tree.a_lookup, (snode, a)) ? tree.n[tree.a_lookup[(snode, a)]] : 0
    # q̂(a) = haskey(tree.a_lookup, (snode, a)) ? normalize_q(tree.q[tree.a_lookup[(snode, a)]]) : normalize_q(estimate_value(planner.value_estimate, mdp, s, 0))
    q̂(a) = haskey(tree.a_lookup, (snode, a)) ? tree.q[tree.a_lookup[(snode, a)]] : estimate_value(planner.value_estimate, mdp, s, 0)
    σ(a) = (c_visit + maximum(tree.n)) * c_scale*q̂(a)
    A = actions(mdp, s)
    valid = valid_action_indices(mdp, s)
    logits = log.(estimate_policy(planner.policy_estimate, mdp, s))
    logits = logits[valid]
    π′ = softmax([logits[i] + σ(a) for (i,a) in enumerate(A)])
    return A[argmax([π′[i] - N(A[i])/(1 + sum(N(b) for b in A)) for i in eachindex(A)])]
end


function valid_action_indices(mdp::MDP, s)
    A = actions(mdp)
    As = actions(mdp, s)

    if length(A) == length(As)
        return eachindex(A)
    else
        idx = Vector{Int}(undef, length(As))
        for (i,a) in enumerate(A)
            for (j,as) in enumerate(As)
                if a == as
                    idx[j] = i
                    break
                end
            end
        end
        return idx
    end
end


"""
Return the best action.

Some publications say to choose action that has been visited the most
e.g., Continuous Upper Confidence Trees by Couëtoux et al.
"""
function best_sanode(tree::GumbelTree, snode::Int)
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
