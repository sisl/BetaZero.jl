using CSV

include("./run_random_test.jl")

test_schedule = CSV.read("./data/tests/random/schedule.csv", CSV.Tables.matrix, header=1)

n = size(test_schedule)[1]
for i=1:n
    println("===== Running configuration $i =====")
    args = test_schedule[i, :]
    run_test(args[1], args[2], args[3], args[4], false)
end
