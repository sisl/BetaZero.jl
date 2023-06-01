using BetaZero
using LightDark
using ParticleFilters
using Plots; default(fontfamily="Computer Modern", framestyle=:box)
using POMDPs
using POMDPTools
using Random
using StatsBase
using ColorSchemes
using POMDPGifs
using POMDPTools
using ProgressMeter
using Reel

function POMDPTools.render(pomdp::LightDarkPOMDP, step;
        steps=nothing,
        network=nothing,
        draw_light_region=true,
        show_belief=true,
        show_belief_traj=true,
        show_obs=true,
        traj_color=:red,
        traj_label="trajectory",
        traj_alpha=0.5,
        traj_lw=2,
        goal_label="goal",
        goal_color=:gold,
        goal_ls=:solid,
        hold=false,
        show_correct=true,
        shared_xmax=nothing,
        colorbar=true,
        use_pgf=false,
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
    xmax = isnothing(shared_xmax) ? length(steps) : shared_xmax

    plotf = hold ? plot! : plot
    if draw_light_region
        plt_lightdark = plotf(xlims=(1, xmax), ylims=(-ymax, ymax), size=(colorbar ? 600 : 550,250), legend=:bottomright, xlabel="time", ylabel="state")
        heatmap!(1:xmax, range(-ymax, ymax, length=50), (x,y)->sqrt(std(observation(pomdp, LightDarkState(0, y)))), c=:grayC, colorbar=colorbar)
        hline!([0], c=goal_color, ls=goal_ls, lw=1, label=goal_label)
    else
        plt_lightdark = plotf()
    end

    show_belief && plot_beliefs(B; hold=true)

    plot!(eachindex(S), Y, c=traj_color, lw=traj_lw, label=use_pgf ? traj_label : false, alpha=traj_alpha)
    !use_pgf && plot!([], [], c=traj_color, lw=traj_lw, label=traj_label) # for better legend

    show_belief_traj && plot!(eachindex(S), Ỹ, c=:blue, lw=1, ls=:dash, label="believed traj.", alpha=0.5)
    show_obs && scatter!(eachindex(S), O, ms=2, c=:cyan, msc=:black, label="observation")

    if isterminal(pomdp, S[end]) && show_correct
        iscorrect = R[end] == pomdp.correct_r
        c = iscorrect ? :lightgreen : :magenta
        hline!([0], c=c, lw=4, label=false)
    end

    return plt_lightdark
end


function gen_lightdark_trajectories(pomdp::LightDarkPOMDP, up::Updater, policies::Vector{<:Policy};
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

function plot_policy_trajectories(policy_simulations, names;
        # [POMCPOW, AdaOPS, BetaZero]
        # colors=[seaborn_light_blue, :gold, seaborn_red],
        # colors=[seaborn_light_blue, :gold, seaborn_red],
        colors=["#ffff00", "#00ffff", "#ff0000"],
        # colors=["#f5831a", "#00ff00", "#ee161f"],
        title="Localization trajectories in \\sc LightDark(10)",
        use_pgf=false)

    if use_pgf
        pgfplotsx()
        plot_config = (;
            grid=false,
            legend=:bottomright,
            legend_font_halign=:left,
            titlefont=15, # 40÷2
            legendfontsize=28÷3,
            guidefontsize=28÷2,
            tickfontsize=28÷2,
            colorbartickfontsizes=28÷2,
            yticks=-10:10:10,
            left_margin=1Plots.mm,
        )
    else
        gr()
        plot_config = (; )
    end

    plot(; title=title, plot_config...)

    ylims!(-15, 15)
    xmax = maximum(map(simulations->maximum(map(length, simulations)), policy_simulations))
    draw_region = true
    for (i,simulations) in enumerate(policy_simulations)
        show_label = true
        for steps in simulations
            render(pomdp, (; t=Inf);
                steps,
                draw_light_region=draw_region, # only draw background on first rendering
                shared_xmax=xmax,
                show_belief=false,
                show_belief_traj=false,
                colorbar=false,
                show_obs=false,
                traj_color=colors[i],
                traj_label=show_label ? names[i] : false,
                traj_alpha=use_pgf ? 0.3 : 0.2,
                traj_lw=1,
                goal_label=false,
                goal_color=:white,
                goal_ls=:dot,
                show_correct=false,
                hold=true,
                use_pgf=use_pgf,
            )
            show_label = false
            draw_region = false
        end
    end
    return plot!(size=(850,250))
end



## Override.
function POMDPGifs.makegif(m::Union{POMDP, MDP}, hist::POMDPTools.Simulators.SimHistory;
                 filename=tempname()*".gif",
                 spec=nothing,
                 show_progress::Bool=true,
                 extra_initial::Bool=false,
                 extra_final::Bool=false,
                 render_kwargs=NamedTuple(),
                 fps::Int=2
                )

    # deal with the spec
    if spec == nothing
        steps = eachstep(hist)
    else
        steps = eachstep(hist, spec)
    end

    if extra_initial
        first_step = first(steps)
        extra_init_step = (t=0, sp=get(first_step, :s, missing), bp=get(first_step, :b, missing))
        steps = vcat(extra_init_step, collect(steps))
    end
    if extra_final
        last_step = last(steps)
        extra_final_step = (t=length(steps)+1, s=get(last_step, :sp, missing), b=get(last_step, :bp, missing), done=true)
        steps = vcat(collect(steps), extra_final_step)
    end

    # create gif
    frames = Frames(MIME("image/png"), fps=fps)
    if show_progress
        p = Progress(length(steps), 0.1, "Rendering $(length(steps)) steps...")
    end

    if haskey(render_kwargs, :include_steps) && render_kwargs[:include_steps]
        render_kwargs_tmp = NamedTuple(filter(p->first(p) != :include_steps, pairs(render_kwargs))) # remove :include_steps keyword
        render_kwargs = merge(render_kwargs_tmp, (; steps=steps))
    end

    for step in steps
        push!(frames, render(m, step; pairs(render_kwargs)...))
        if show_progress
            next!(p)
        end
    end
    if show_progress
        @info "Creating Gif..."
    end
    write(filename, frames)
    if show_progress
        @info "Done Creating Gif."
    end
    return POMDPGifs.SavedGif(filename)
end

