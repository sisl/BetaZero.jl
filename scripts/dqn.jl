using Revise
using Crux
using Flux
using POMDPGym

include(joinpath(@__DIR__, "representation_minex.jl"))

mdp = BeliefMDP(pomdp, up, simple_minex_belief_reward)
as = actions(mdp)
S = state_space(mdp)
sdim = Crux.dim(S)

A() = DiscreteNetwork(Chain(
    Conv((5,5), sdim[end]=>64, relu),
    Conv((5,5), 64=>128, relu),
    Flux.flatten,
    Dense(prod([sdim[1] - 2*4, sdim[2] - 2*4, 128])=>256, relu),
    Dense(256=>256, relu),
    Dense(256=>256, relu),
    Dense(256=>length(as))), as)

solver_dqn = DQN(π=A(), S=S, N=2000)
@time π_dqn = solve(solver_dqn, mdp)
p = plot_learning(solver_dqn, title="Training Curve")

nothing # REPL
