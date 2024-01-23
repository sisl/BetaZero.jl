is_constrained = true
resume = false
if resume
    policy = solve(solver, pomdp; surrogate=policy.surrogate, resume)
else
    include("launch_spillpoint.jl")
    include("solver_spillpoint.jl")
    policy = solve(solver, pomdp)
end

filename_suffix = "spillpoint_cbz"
BetaZero.save_policy(policy, "policy_$filename_suffix.bson")
BetaZero.save_solver(solver, "solver_$filename_suffix.bson")

#=
solver.mcts_solver.final_criterion = MCTS.MaxZQN(zq=solver.mcts_solver.final_criterion.zq, zn=solver.mcts_solver.final_criterion.zn)
policy = solve_planner!(solver, policy.surrogate)
default(fontfamily="Computer Modern", framestyle=:box)
@info "Number of processes: $(nworkers())"
@info "Number of threads: $(Threads.nthreads())"
Random.seed!(1)
ds0 = initialstate(pomdp)
b0 = initialize_belief(up, ds0)
s0 = rand(b0)
A = actions(pomdp, s0)
# policy = RandomPolicy(pomdp)
# policy = FunctionPolicy(b->isnothing(particles(b)[1].x_inj) ? rand(A) : (:inject, 0.07))
policy = FunctionPolicy(b->rand(setdiff(actions(pomdp, b), [(:stop, 0.0)])))
@time for (s, a, r, sp, t, b) in stepthrough(pomdp, policy, up, b0, "s,a,r,sp,t,b", max_steps=25)
    @show t, a, r
    render(pomdp, s, a; belief=b, timestep=t) |> display
end
=#

nothing # REPL