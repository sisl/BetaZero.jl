using POMDPGifs

include("representation_lightdark.jl")
include("plot_lightdark.jl")

Random.seed!(0xC0FFEE)
sim = GifSimulator(filename="lightdark.gif",
    max_steps=100,
    render_kwargs=(;
        include_steps=true,
        network=policy.surrogate,
        show_belief=false,
        show_obs=false,
        show_belief_traj=true,
    )
)

policy2run = :betazero
# policy2run = :pomcpow
# policy2run = :adaops
# policy2run = :mcts
# policy2run = :rawpolicy

if policy2run == :betazero
    gif_planner = policy
elseif policy2run == :pomcpow
    if !@isdefined(pomcpow_planner)
        # Avoid rerunning value iteration bounds...
        pomcpow_planner = solve_pomcpow(pomdp, nothing; override=true)
    end
    gif_planner = pomcpow_planner
elseif policy2run == :adaops
    if !@isdefined(adaops_planner)
        # Avoid rerunning value iteration bounds...
        adaops_planner = solve_adaops(pomdp)
    end
    gif_planner = adaops_planner
elseif policy2run == :mcts
    zero_mcts_planner = extract_mcts(solver, pomdp)
    gif_planner = zero_mcts_planner
elseif policy2run == :rawpolicy
    raw_policy = RawNetworkPolicy(pomdp, policy.surrogate)
    gif_planner = raw_policy
else
    error("No policy specified for $policy2run")
end

simulate(sim, pomdp, gif_planner, up)