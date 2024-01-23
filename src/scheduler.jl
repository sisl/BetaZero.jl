struct LearningRateScheduler
    lrs::Vector
    iterations::Vector
end

get_learning_rate(lr::Float64, ::Int) = lr
function get_learning_rate(scheduler::LearningRateScheduler, iteration::Int)
    iterations = vcat(0, scheduler.iterations)
    for i in 2:length(iterations)
        if iterations[i-1] ≤ iteration < iterations[i]
            return scheduler.lrs[i-1]
        end
    end
    return scheduler.lrs[end]
end
