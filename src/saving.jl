"""
Save performance metrics to a file.
"""
function save_metrics(solver::BetaZeroSolver, filename::String)
    metrics = solver.performance_metrics
    BSON.@save filename metrics
end


"""
Save policy to file (MCTS planner and surrogate objects together).
"""
function save_policy(policy::BetaZeroPolicy, filename::String)
    BSON.@save "$filename" policy
end


"""
Load policy from file (MCTS planner and surrogate objects together).
"""
function load_policy(filename::String)
    BSON.@load "$filename" policy
    return localize_policy(policy)
end


"""
Handle anonymous function serialization issues
"""
function localize_policy(policy::BetaZeroPolicy)
    network = policy.surrogate
    normalize_input = policy.parameters.nn_params.normalize_input
    normalize_output = policy.parameters.nn_params.normalize_output

    if normalize_input
        infunc = network.layers[1]
        mean_x = hasproperty(infunc.mean_x, :contents) ? infunc.mean_x.contents : infunc.mean_x
        std_x = hasproperty(infunc.std_x, :contents) ? infunc.std_x.contents : infunc.std_x
        ϵ_std = infunc.ϵ_std
        normalize_x = x -> (x .- mean_x) ./ (std_x .+ ϵ_std)
        heads = network.layers[end]
    end

    if normalize_output
        vhf = network.layers[end].layers.value_head.layers[end]
        mean_y = hasproperty(vhf.mean_y, :contents) ? vhf.mean_y.contents : vhf.mean_y
        std_y = hasproperty(vhf.std_y, :contents) ? vhf.std_y.contents : vhf.std_y

        unnormalize_y = y -> (y .* std_y) .+ mean_y

        heads = network.layers[end]
        value_head = heads.layers.value_head
        value_head = Chain(value_head.layers[1:end-1]..., unnormalize_y) # NOTE end-1 to remove it first.
        policy_head = heads.layers.policy_head
        pfail_head = heads.layers.pfail_head
        heads = Parallel(heads.connection, value_head=value_head, policy_head=policy_head, pfail_head=pfail_head)
    end

    if normalize_input
        network = Chain(normalize_x, network.layers[2:end-1]..., heads)
    end

    if normalize_output
        network = Chain(network.layers[1:end-1]..., heads)
    end

    policy.surrogate = network

    solve_planner!(policy)

    return policy
end


"""
Save just the surrogate model to a file.
"""
save_surrogate(policy::BetaZeroPolicy, filename::String)
function save_surrogate(surrogate::Surrogate, filename::String)
    BSON.@save "$filename" surrogate
end


"""
Load just the surrogate model from a file.
"""
function load_surrogate(filename::String)
    BSON.@load "$filename" surrogate
    return surrogate
end


"""
Save the solver to a file.
"""
function save_solver(solver::BetaZeroSolver, filename::String)
    BSON.@save "$filename" solver
end


"""
Load the solver from a file.
"""
function load_solver(filename::String)
    BSON.@load "$filename" solver
    return solver
end


"""
Save off policy during BetaZero iterations.
"""
function incremental_save(solver::BetaZeroSolver, f::Surrogate, i="")
    if solver.nn_params.incremental_save
        filename = solver.nn_params.policy_filename
        file, ext = splitext(filename)
        filename = string(file, "_$i", ext)

        policy = solve_planner!(solver, f)
        parameters = policy.parameters
        surrogate = policy.surrogate
        cache = (surrogate, parameters)

        BSON.@save "$filename" cache
    end
end


"""
Load incremental network. Return (; network, parameters)
"""
function load_incremental(filename::String)
    BSON.@load "$filename" cache
    network = cache[1]
    parameters = cache[2]

    # Handle anonymous function serialization issues
    normalize_input = parameters.nn_params.normalize_input
    normalize_output = parameters.nn_params.normalize_output

    if normalize_input
        infunc = network.layers[1]
        mean_x = infunc.mean_x.contents
        std_x = infunc.std_x.contents
        ϵ_std = infunc.ϵ_std
        normalize_x = x -> (x .- mean_x) ./ (std_x .+ ϵ_std)
        heads = network.layers[end]
    end

    if normalize_output
        vhf = network.layers[end].layers.value_head.layers[end]
        mean_y = vhf.mean_y.contents
        std_y = vhf.std_y.contents

        unnormalize_y = y -> (y .* std_y) .+ mean_y

        heads = network.layers[end]
        value_head = heads.layers.value_head
        value_head = Chain(value_head.layers[1:end-1]..., unnormalize_y) # NOTE end-1 to remove it first.
        policy_head = heads.layers.policy_head
        pfail_head = heads.layers.pfail_head
        heads = Parallel(heads.connection, value_head=value_head, policy_head=policy_head, pfail_head=pfail_head)
    end

    if normalize_input
        network = Chain(normalize_x, network.layers[2:end-1]..., heads)
    end

    if normalize_output
        network = Chain(network.layers[1:end-1]..., heads)
    end

    return (; network, parameters)
end


"""
For backwards compatability with policies saved before commit c0cff08 on June 1, 2023.
(i.e., input `accuracy_func` was replaced with `BetaZero.accuracy` interface function).
"""
function BSON.newstruct!(::BetaZeroSolver, args...)
    if length(args) == 23
        fields = []
        i = 1
        for arg in args
            # NOTE: Skip 17th argument which was `accuracy_func` and was replaced with `BetaZero.accuracy`
            is_accuracy_func = (i == 17)
            if typeof(arg) == Vararg
                if !is_accuracy_func
                    push!(fields, arg[i]) # unroll Vararg element
                end
            else
                if !is_accuracy_func
                    push!(fields, arg)
                end
            end
            i = i + 1
        end
        return BetaZeroSolver(fields...)
    else
        return BetaZeroSolver(args...)
    end
end

Base.convert(::Type{AbstractMCTSSolver}, mcts_solver::AbstractMCTSSolver) = mcts_solver
