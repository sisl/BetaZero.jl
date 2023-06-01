using Revise
using MinEx
using POMDPs
using POMCPOW
using BetaZero
using Reel
using ProgressMeter

include("representation_minex.jl")
include("minex_tikz.jl")

Base.isless(x::Symbol, ::Tuple{Int64, Int64}) = true
Base.isless(::Tuple{Int64, Int64}, ::Symbol) = false

pomdp = MinExPOMDP(; n_particles=2_500, σ_abc=0.15)
up = ParticleHistoryBeliefUpdater(BootstrapFilter(pomdp, pomdp.n_particles))

over_mean = true
rerun_pomcpow = false

if rerun_pomcpow
    Random.seed!(0xC0FFEE)
    ds0 = initialstate(pomdp)
    s0 = rand(ds0)
    b0 = initialize_belief(up, ds0)

    pomcpow_solver = POMCPOWSolver(tree_queries=1_000_000,
        criterion=POMCPOW.MaxUCB(100.0),
        k_action=4.0,
        alpha_action=0.5,
        k_observation=2.0,
        alpha_observation=0.25,
        estimate_value=0.0,
        tree_in_info=true,
        max_depth=38,
    )
    pomcpow_planner = solve(pomcpow_solver, pomdp)
    @time pomcpow_steps = collect(stepthrough(pomdp, pomcpow_planner, up, b0, s0, "sp,a,r,b,bp,t,action_info", max_steps=1))

    tree = pomcpow_steps[1].action_info[:tree]
    root_children_indices = tree.tried[1]
    root_actions = tree.a_labels[root_children_indices]
    root_counts = tree.n[root_children_indices]
    root_values = tree.v[root_children_indices]

    idx = sortperm(root_actions)
    root_actions = root_actions[idx]
    root_counts = root_counts[idx]
    root_values = root_values[idx]

    N_pomcpow = root_counts ./ sum(root_counts)
    Q_pomcpow = root_values

    p_pomcpow = softmax(Q_pomcpow) # POMCPOW uses MaxQ

    write_tikz_datatable(root_actions[3:end], p_pomcpow[3:end], "drillspomcpow.dat"; xmax=pomdp.grid_dims[1], ymax=pomdp.grid_dims[2], scale=50, dir=".")

    p_decisions = p_pomcpow[1:2]
    write_tikz_decisiontable(p_decisions, "decisionspomcpow.dat"; dir=".")

    b̃ = BetaZero.input_representation(pomcpow_steps[1].b)
    if over_mean
        bmap = b̃[:,:,1] # mean
    else
        bmap = b̃[:,:,2] # std
    end
    write_tikz_datatable([(i,j) for i in 1:pomdp.grid_dims[1], j in 1:pomdp.grid_dims[2]], bmap, "mapstdpomcpow.dat"; dir=".")
end

reload_betazero = true
rerun_betazero = true
compile_tikz = true
num_steps = 15

if reload_betazero
    # BetaZero policy
    # policy = load_policy("final_policy_mineral_exploration.bson")
    # solver = load_solver("final_solver_mineral_exploration.bson")

    solver.mcts_solver.final_criterion.τ = 0 # Important that it's the sovler
    solver.mcts_solver.final_criterion.zq = 0.5
    solver.mcts_solver.final_criterion.zn = 0.5
    # solver.mcts_solver.init_Q = bootstrap(policy.surrogate)
    # policy = solve_planner!(solver, policy_checkpoint.network)

    # TODO: apply_bootstrap! function
    policy = solve_planner!(solver, policy.surrogate)
    raw_policy = RawNetworkPolicy(pomdp, policy.surrogate)
end


Random.seed!(12345) # Appendix (0xC0FFEE, 1, 9, 1000, 555)
ds0 = initialstate(pomdp)
s0 = rand(ds0)
b0 = initialize_belief(up, ds0)


if rerun_betazero
    local plt
    local truth
    V = []
    R = []
    belief_volume = []
    belief_volume_stderr = []
    frames = Frames(MIME("image/png"), fps=1)
    global max_time = 0

    @time for step in stepthrough(pomdp, policy, up, b0, s0, "s,a,r,b,sp,bp,t,action_info", max_steps=num_steps)
        t = step.t
        global max_time = t
        @info "Step $t..."
        counts = step.action_info[:counts]
        local root_actions = collect(keys(counts))
        local root_counts_and_values = collect(values(counts))
        local root_counts = first.(root_counts_and_values)
        local root_values = last.(root_counts_and_values)

        local idx = sortperm(root_actions)
        root_actions = root_actions[idx]
        root_counts = root_counts[idx]
        root_values = root_values[idx]


        zq = policy.planner.solver.final_criterion.zq
        zn = policy.planner.solver.final_criterion.zn
        QN = (softmax(root_values).^zq .* root_counts.^zn)
        tree_P = normalize(QN, 1)

        action_space = POMDPs.actions(pomdp)
        ϵ = 1e-8
        P = ϵ * ones(length(action_space))

        for (i,a′) in enumerate(action_space)
            j = findfirst(tree_a->tree_a == a′, root_actions)
            if !isnothing(j)
                P[i] = tree_P[j]
            end
        end

        P = normalize(P, 1)
        write_tikz_datatable(action_space[3:end], P[3:end], "drills$(t).dat"; xmax=pomdp.grid_dims[1], ymax=pomdp.grid_dims[2], scale=50, dir=".")

        display(BetaZero.barplot(action_space, P))

        local p_decisions = P[1:2]
        write_tikz_decisiontable(p_decisions, "decisions$(t).dat"; dir=".")

        local b̃ = BetaZero.input_representation(step.b)
        local μ = b̃[:,:,1] # mean
        local σ = b̃[:,:,2] # std
        local bmap = over_mean ? μ : σ
        write_tikz_datatable([(i,j) for i in 1:pomdp.grid_dims[1], j in 1:pomdp.grid_dims[2]], bmap, "mapstd$(t).dat"; dir=".")
        write_tikz_actiontable(step.a, "action$(t).dat"; dir=".")

        push!(R, step.r)
        v = value_lookup(policy, step.b)
        push!(V, v)

        truth = extraction_reward(pomdp, step.s)

        economical_volume = [extraction_reward(pomdp, s) for s in particles(step.b)]
        b_mean, b_stderr = mean_and_stderr(economical_volume)
        push!(belief_volume, b_mean)
        push!(belief_volume_stderr, b_stderr)

        @info v, belief_volume[end], truth

        # bz_beliefs = map(step->step.b, bz_steps)
        # bz_actions = map(step->step.a, bz_steps)
        # tikz_policy_plots(pomdp, policy, bz_beliefs, bz_actions)

        plt = plot_belief(step.b, step.sp; a=step.a, scale=1)

        push!(frames, plt)
        Plots.savefig("minex_belief$(step.t).png")

        BSON.@save "minex_belief_mean$(step.t).bson" μ
        BSON.@save "minex_belief_std$(step.t).bson" σ
        local drill_action_x
        local drill_action_y
        if !isempty(step.sp.drill_locations)
            xloc = map(last, step.sp.drill_locations) # Note y-first, x-last
            yloc = map(first, step.sp.drill_locations)
            drill_action_x = xloc
            drill_action_y = yloc
            drill_actions = (drill_action_x, drill_action_y)
            BSON.@save "minex_drill_actions$(step.t).bson" drill_actions
        end
    end
    [push!(frames, plt) for _ in 1:5] # duplicate last frame
    write("minex_belief.gif", frames)

    plot(belief_volume, ribbon=belief_volume_stderr, fillalpha=0.1, label="estimated volume", c=:darkgreen, size=(600,250), xlabel="iteration", ylabel="economical volume", marker=:circle, msc=:darkgreen, mc=:white, ms=2, margin=2Plots.mm)
    hline!([truth], label="true volume", c=:black, ls=:dash)
    # γ = discount(pomdp)
    # G = BetaZero.compute_returns(R; γ)
    # plot!(G, label="true returns", c=:black, ls=:dash)
    Plots.savefig("minex_values.png")

    if compile_tikz
        tikz_dir = joinpath(@__DIR__, "..", "tex")
        tikz_data_dir = joinpath(tikz_dir, "data")

        @info "Compiling TiKZ..."
        @showprogress for t in 1:max_time
            cp("decisions$(t).dat", joinpath(tikz_data_dir, "decisions.dat"), force=true)
            cp("mapstd$(t).dat", joinpath(tikz_data_dir, "map.dat"), force=true)
            cp("drills$(t).dat", joinpath(tikz_data_dir, "drills.dat"), force=true)
            cp("action$(t).dat", joinpath(tikz_data_dir, "action.dat"), force=true)

            @suppress_out cd(tikz_dir) do
                run(`pdflatex policy_map.tex`)
                mv("policy_map.pdf", "policy_map$(t).pdf", force=true)
            end
        end
        @info "Done."
    end
end
