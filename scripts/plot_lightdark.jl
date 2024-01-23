# using Revise
# using BetaZero
# using LightDark
# using ParticleFilters
# using ParticleBeliefs
using Plots; default(fontfamily="Computer Modern", framestyle=:box)
# using POMDPs
# using POMDPTools
# using Random
# using StatsBase
# using ColorSchemes
# using POMDPGifs
# using POMDPTools
using ProgressMeter
# using Reel

# backwards compat.
# lightdark_accuracy_func(pomdp, b0, s0, states, actions, returns) = returns[end] == pomdp.correct_r
# lightdark_belief_reward(pomdp, b, a, bp) = mean(reward(pomdp, s, a) for s in ParticleFilters.particles(b))

# function POMDPTools.render(pomdp::LightDarkPOMDP, step;
function POMDPTools.render(pomdp::POMDP, step;
        steps=nothing,
        network=nothing,
        draw_light_region=true,
        show_belief=true,
        show_belief_traj=true,
        show_obs=true,
        show_goal=true,
        palette=:viridis,
        traj_color=:red,
        traj_label="trajectory",
        traj_alpha=0.5,
        traj_lw=2,
        mark_color=:black,
        goal_label="goal",
        goal_color=:white,
        goal_bound_color=:gray,
        goal_ls=:dash,
        goal_bound_ls=:dash,
        goal_lw=1,
        hold=false,
        show_correct=false,
        show_failure=true,
        shared_xmax=nothing,
        colorbar=true,
        use_pgf=false,
        xmax=-Inf,
    )
    S = [steps[1].s]
    A = [0.0]
    O = [0.0]
    B = [steps[1].b]
    R = [0.0]
    rd = x->round(x, digits=4)

    for localstep in steps
        if localstep.t > step.t
            break
        end
        push!(S, localstep.sp)
        push!(A, localstep.a)
        push!(O, localstep.o)
        push!(B, localstep.bp)
        push!(R, localstep.r)
        # ỹ, σ = mean_and_std(s.y for s in ParticleFilters.particles(b))
    end

    # if hasproperty(policy, :surrogate)
    #     G = BetaZero.compute_returns(R; γ=POMDPs.discount(pomdp))
    #     Ṽ = [Float64(BetaZero.value_lookup(policy.surrogate, b)) for b in B]
    #     value_mae = mean(abs.(G .- Ṽ))
    #     @info "Value estimate MAE: $(round(value_mae, digits=4))"
    # end

    function plot_beliefs(B; hold=false)
        !hold && plot()
        for i in eachindex(B)
            local n = length(ParticleFilters.particles(B[i]))
            local P = ParticleFilters.particles(B[i])
            local X = i*ones(n)
            local Y = [p.y for p in P]
            scatter!(X, Y, c=:black, msc=:white, alpha=0.25, ms=2, label=i==1 ? "belief" : false)
        end
        return plot!()
    end

    Y = map(s->s.y, S)
    Ỹ = map(b->mean(p.y for p in ParticleFilters.particles(b)), B)
    ymax = max(15, max(maximum(Y), abs(minimum(Y))))*1.1
    xmax = max(isnothing(shared_xmax) ? length(steps) : shared_xmax, xmax)

    plotf = hold ? plot! : plot
    if draw_light_region
        plt_lightdark = plotf(xlims=(1, xmax), ylims=(-ymax, ymax), size=(colorbar ? 600 : 550,250), legend=:bottomright, xlabel="time", ylabel="state")
        heatmap!(1:xmax, range(-ymax, ymax, length=50), (x,y)->sqrt(std(observation(pomdp, LightDarkState(0, y, 0)))), c=:grayC, colorbar=colorbar)
    else
        plt_lightdark = plotf()
    end

    show_belief && plot_beliefs(B; hold=true)

    plot!(eachindex(S), Y, palette=palette, c=traj_color, lw=traj_lw, label=use_pgf ? traj_label : false, alpha=traj_alpha)
    show_end_failure = show_failure && isfailure(pomdp, S[end], A[end])
    end_mark = show_end_failure ? :star4 : :circle
    end_msc = show_end_failure ? :red : mark_color

    scatter!(eachindex(S)[end:end], Y[end:end], palette=palette, c=traj_color, mark=end_mark, msc=end_msc, ms=4, msw=0.5, label=false, alpha=1.5*traj_alpha)
    # scatter!(eachindex(S), Y, palette=palette, c=traj_color, ms=2, msw=1, msc=:black, label=false, alpha=traj_alpha)
    !use_pgf && plot!([], [], palette=palette, c=traj_color, lw=traj_lw, label=traj_label) # for better legend

    show_belief_traj && plot!(eachindex(S), Ỹ, c=:blue, lw=1, ls=:dash, label="believed traj.", alpha=0.5)
    show_obs && scatter!(eachindex(S), O, ms=2, c=:cyan, msc=:black, label="observation")

    if show_goal
        Plots.hline!([1], c=goal_bound_color, ls=goal_bound_ls, lw=goal_lw, label=goal_label)
        Plots.hline!([0], c=goal_color, ls=goal_ls, lw=goal_lw, label=goal_label)
        Plots.hline!([-1], c=goal_bound_color, ls=goal_bound_ls, lw=goal_lw, label=goal_label)
    end

    if isterminal(pomdp, S[end]) && show_correct
        iscorrect = !isfailure(pomdp, S[end], A[end])
        c = iscorrect ? :lightgreen : :magenta
        Plots.hline!([0], c=c, lw=4, label=false)
    end

    return plt_lightdark
end


# function gen_lightdark_trajectories(pomdp::LightDarkPOMDP, up::Updater, policies::Vector{<:Policy};
function gen_lightdark_trajectories(pomdp::POMDP, up::Updater, policies::Vector{<:Policy};
        n_sims::Int=10,
        max_steps=100)

    policy_simulations = []
    for (i,policy) in enumerate(policies)
        progress = Progress(n_sims)
        channel = RemoteChannel(()->Channel{Bool}(), 1)

        @async while take!(channel)
            next!(progress)
        end

        @info typeof(policy)
        simulations = pmap(sim->begin
            steps = simulate(HistoryRecorder(max_steps=max_steps), pomdp, policy, up)
            put!(channel, true) # trigger progress bar update
            steps
        end, 1:n_sims)
        put!(channel, false) # tell printing task to finish

        push!(policy_simulations, simulations)
    end

    return policy_simulations
end

# seaborn_deep
seaborn_green = "#55a868"
seaborn_blue = "#4c72b0"
seaborn_light_blue = "#64b5cd"
seaborn_red = "#c44e52"
seaborn_purple = "#8172b2"

function plot_policy_trajectories(pomdp, policy_simulations, names;
        # [POMCPOW, AdaOPS, BetaZero]
        # colors=[seaborn_light_blue, :gold, seaborn_red],
        # colors=[seaborn_light_blue, :gold, seaborn_red],
        # colors=["#ffff00", "#00ffff", "#ff0000"],
        # colors=["#ffff00", "#00ff99", "#ff4400"],
        # colors=["#FEDD5C", "#6FC3FF", "#E04F39"],
        # colors=["#1AECBA", "#ffff00", "#ff0000"], # E50808
        # colors=["#ffff00", "#ff0000", "#0088ff"],
        # colors=["#FEDD5C", seaborn_red, "#00548f"],
        # colors=["#FEDD5C", seaborn_red, "#25838e"],
        # colors=["#fde725", seaborn_red, "#25838e"],
        # colors=["#35b778", "#fde725", "#482878"],
        # colors=["#fde725", "#35b778", "#482878"],
        colors=["#FDE725", "#21918C", "#440154"],
        mark_colors=[:black, :black, :lightgray],
        # mark_colors=["#290132", "#145754", "#290132"], # https://mdigi.tools/darken-color/ 40%
        # mark_colors=["#fdec51", "#2dc7c0", "#8802a8"], # https://mdigi.tools/lighten-color/ 20%
        # colors=["#f5831a", "#00ff00", "#ee161f"],
        title=raw"Localization trajectories in \textsc{LightDark}$(10)$",
        use_pgf=false, xmax=missing, kwargs...)

    if use_pgf
        pgfplotsx()
        plot_config = (;
            grid=false,
            legend=:bottomright,
            legend_font_halign=:left,
            titlefont=20, # 40÷2
            legendfontsize=28÷2,
            guidefontsize=28÷1.5,
            tickfontsize=28÷1.5,
            colorbartickfontsizes=28÷1.5,
            yticks=-10:10:10,
            left_margin=1Plots.mm,
            fontfamily="Times",
        )
    else
        gr()
        plot_config = (; )
    end

    plot(; title=title, plot_config...)

    ylims!(-15, 15)
    if ismissing(xmax)
        xmax = maximum(map(simulations->maximum(map(length, simulations)) + 2, policy_simulations))
    end

    draw_region = true
    for (i,simulations) in enumerate(policy_simulations)
        show_label = true
        for (t,steps) in enumerate(simulations)
            render(pomdp, (; t=Inf);
                steps,
                draw_light_region=draw_region, # only draw background on first rendering
                shared_xmax=xmax,
                show_belief=false,
                show_belief_traj=false,
                colorbar=false,
                show_obs=false,
                # traj_color=colors[i],
                # mark_color=mark_colors[i],
                traj_color=t,
                mark_color=t,
                palette=:jet,
                traj_label=show_label ? names[i] : false,
                traj_alpha=use_pgf ? 0.5 : 0.5,
                traj_lw=2,
                show_goal=(i == length(policy_simulations) && t == length(simulations)),
                goal_label=false,
                show_correct=false,
                show_failure=true,
                hold=true,
                use_pgf=use_pgf,
                kwargs...
            )
            show_label = false
            draw_region = false
        end
    end
    return plot!(size=(700,250))
end

plot_predicted_failure(pomdp::POMDP, policy::BetaZeroPolicy) = plot_predicted_failure(pomdp, policy.surrogate)

function plot_predicted_failure(pomdp::POMDP, surrogate::Surrogate)
    xrange = range(0, 5, length=50)
    yrange = range(-1, 12, length=50)

    plot_config = (size=(500,300), xlabel="belief std", ylabel="belief mean")

    # t = 0 # pomdp.max_time - 1
    T = 0:10:pomdp.max_time
    # T = pomdp.max_time:pomdp.max_time
    include_time = false
    discrete_actions = true
    max_p_fail = false
    full_range_pfail_colors = false

    if include_time
        Yv = (x,y)->mean(surrogate(Float32.([x y t])')[1] for t in T)  # last [1] for [y, t]
    else
        Yv = (x,y)->surrogate(Float32.([x y])')[1]
    end
    Yvdata = [Yv(y,x) for y in yrange, x in xrange]
    cmap_value = BetaZero.shifted_colormap(Yvdata; colors=BetaZero.VALUE_PALETTE)
    plt_value = Plots.heatmap(xrange, yrange, Yvdata; cmap=cmap_value, title="value estimate", plot_config...)

    as = actions(pomdp)
    if include_time
        if discrete_actions
            Yπ = (x,y)->as[mode(argmax(surrogate(Float32.([x y t])')[2:end-1]) for t in T)]
            # Yπ = (x,y)->as[round(Int, mean(argmax(surrogate(Float32.([x y t])')[2:end-1]) for t in T))]
        else
            Yπ = (x,y)->mean(as[argmax(surrogate(Float32.([x y t])')[2:end-1])] for t in T)
        end
        levels = 5
    else
        Yπ = (x,y)->as[argmax(surrogate(Float32.([x y])')[2:end-1])]
        levels = 2
    end
    Yπdata = [Yπ(y,x) for y in yrange, x in xrange]
    plt_policy = Plots.heatmap(xrange, yrange, Yπdata; cmap=BetaZero.POLICY_COLOR, levels=levels, lw=0, title="policy estimate", plot_config...)

    if include_time
        if max_p_fail
            Ys = (x,y)->maximum(surrogate(Float32.([x y t])')[end] for t in T)
        else
            Ys = (x,y)->mean(surrogate(Float32.([x y t])')[end] for t in T)
        end
    else
        Ys = (x,y)->surrogate(Float32.([x y])')[end]
    end
    if full_range_pfail_colors
        pfail_args = (clims=(0,1),)
    else
        pfail_args = ()
    end
    Ysdata = [Ys(y,x) for y in yrange, x in xrange]
    plt_pfail = Plots.contourf(xrange, yrange, Ysdata; cmap=:jet, pfail_args..., levels=15, lw=0.1, title="predicted probability of failure", plot_config...)
    
    plot(plt_value, plt_policy, plt_pfail, layout=(3,1), size=(400, 700), leftmargin=5Plots.mm)
end


## Override.
# function POMDPGifs.makegif(m::Union{POMDP, MDP}, hist::POMDPTools.Simulators.SimHistory;
#                  filename=tempname()*".gif",
#                  spec=nothing,
#                  show_progress::Bool=true,
#                  extra_initial::Bool=false,
#                  extra_final::Bool=false,
#                  render_kwargs=NamedTuple(),
#                  fps::Int=2
#                 )

#     # deal with the spec
#     if spec == nothing
#         steps = eachstep(hist)
#     else
#         steps = eachstep(hist, spec)
#     end

#     if extra_initial
#         first_step = first(steps)
#         extra_init_step = (t=0, sp=get(first_step, :s, missing), bp=get(first_step, :b, missing))
#         steps = vcat(extra_init_step, collect(steps))
#     end
#     if extra_final
#         last_step = last(steps)
#         extra_final_step = (t=length(steps)+1, s=get(last_step, :sp, missing), b=get(last_step, :bp, missing), done=true)
#         steps = vcat(collect(steps), extra_final_step)
#     end

#     # create gif
#     frames = Frames(MIME("image/png"), fps=fps)
#     if show_progress
#         p = Progress(length(steps), 0.1, "Rendering $(length(steps)) steps...")
#     end

#     if haskey(render_kwargs, :include_steps) && render_kwargs[:include_steps]
#         render_kwargs_tmp = NamedTuple(filter(p->first(p) != :include_steps, pairs(render_kwargs))) # remove :include_steps keyword
#         render_kwargs = merge(render_kwargs_tmp, (; steps=steps))
#     end

#     for step in steps
#         push!(frames, render(m, step; pairs(render_kwargs)...))
#         if show_progress
#             next!(p)
#         end
#     end
#     if show_progress
#         @info "Creating Gif..."
#     end
#     write(filename, frames)
#     if show_progress
#         @info "Done Creating Gif."
#     end
#     return POMDPGifs.SavedGif(filename)
# end

