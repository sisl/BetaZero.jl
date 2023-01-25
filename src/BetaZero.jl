module BetaZero

# function policy_iteration()
#     policy_evaluation()
#     policy_improvement()
# end

# function policy_evaluation() end
# function policy_improvement() end

"""
The main BetaZero policy iteration algorithm.
"""
function betazero(; interations=10)
    f_prev = nothing
    data = nothing

    for i in 1:interations
        # 1) Optimize neural network parameters with recent simulated data.
        f_curr = train_network(data)

        # 2) BetaZero agent is evaluated (compared to previous agent, beating it in 55%+ simulations).
        f = evaluate_agent(f_prev, f_curr)

        # 3) Generate new data using the best BetaZero agent so far.
        data = generate_data(f)
    end

    return β
end


function train_network()
    # TODO: Include code from `cnn.jl`
        # TODO: Include action/policy vector
        # TODO: Change loss to include CE-loss

    # return f
end


function evaluate_agent(f_prev, f_curr; simulations=100, iterations=50)
    prev_correct = 0
    curr_correct = 0

    # Run a number of simulations to evaluate the two neural networks (`f_prev` and `f_curr`)
    for i in 1:simulations
        ans_prev = evaluate(f_prev; iterations) # TODO.
        ans_curr = evaluate(f_curr; iterations) # TODO.
        ans_true = truth() # TODO.
        if ans_prev == ans_true
            prev_correct += 1
        end
        if ans_curr == ans_true
            curr_correct += 1
        end
    end

    if curr_correct >= prev_correct
        return f_curr
    else
        return f_prev
    end
end


function generate_data(f)
    # Run MCTS or POMCPOW to generate data using the neural network `f`
    # TODO. See `run_trial` in utils (reuse MixedFidelityModelSelection to run in parallel?)
        # TODO: See `betazero.jl` in MEParallel.jl

    # return data
end


function collect_metrics()
    # TODO: Collect intermediate results from the steps above.
end


end # module
