function rocksample_network_viz(pomdp::POMDP, up::Updater, policy::Policy)
    ds0 = initialstate(pomdp)
    s0 = rand(ds0)
    s0 = RSState(RSPos(1,1), s0.rocks)
    b0 = initialize_belief(up, ds0)
    f = policy.surrogate
    V, P = rocksample_value_and_policy(f, b0)
    rocksample_network_viz(pomdp, V, P)
end


function rocksample_network_viz(pomdp::POMDP, policy::Policy, step; include_policy=true, include_agent=true, use_pgf=false, title="RockSample value network")
    if use_pgf
        pgfplotsx()
    else
        gr()
    end

    plot(title=title,
         legend=false,
         titlefont=15, # 40÷2
         legendfontsize=28÷2,
         guidefontsize=28÷2,
         tickfontsize=28÷2,
         colorbartickfontsizes=28÷2,
         framestyle=:box,
         grid=false,
         widen=false,
    )

    plot!(xticks=false, yticks=false)

    f = policy.surrogate
    b = step.b
    V, P = rocksample_value_and_policy(f, b)
    plt_value = rocksample_network_viz(pomdp, V, P; hold=true, include_rocks=false)

    good_rocks = pomdp.rocks_positions[step.s.rocks]

    for rocks in pomdp.rocks_positions
        if rocks in good_rocks
            c = :green
        else
            c = :red
        end
        scatter!([rocks[1]], [rocks[2]], label=false, c=c, msc=:white, ms=3, widen=false)
    end

    if include_agent
        scatter!([step.s.pos[1]], [step.s.pos[2]], c=:orange, msc=:black, ms=3, marker=:square, alpha=0.5, label=false)
    end

    if include_policy
        input = network_input(b)
        p = f(input)[2:end]
        policy_colors = vcat(:black, fill(:orange,4), fill(:gray, 15))
        policy_colors[step.a] = :crimson
        plt_policy = rocksample_policy_plot(p; colors=policy_colors)
        scatter!([step.a], [p[step.a]], c=:crimson, msc=:black, ms=5, marker=:dtriangle, label=false)
        return plot!(plt_policy, plt_value, size=(400, 600), layout=@layout[a; b{0.7h}])
    else
        return plot!(plt_value, size=(400,400))
    end
end


function rocksample_network_viz(pomdp, V::Matrix, P::Matrix; hold=false, include_rocks=true)
    colors = vcat(:black, fill(:orange,4), fill(:gray, 15))
    # Plots.heatmap(P, ratio=1, label=false, c=colors)

    value_color = :viridis # cgrad(["#FFFFFF", "#8C1515"], rev=false)
    heatmapf = hold ? Plots.heatmap! : Plots.heatmap
    heatmapf(V, ratio=1, label=false, c=value_color)
    xlims!(ylims()...)

    if include_rocks
        for (i,rocks) in enumerate(pomdp.rocks_positions)
            scatter!([rocks[1]], [rocks[2]], label=false, c=:black, msc=:white, ms=3, widen=false)
        end
    end

    plot!()
end


function rocksample_value_and_policy(f, b)
    V = Matrix(undef, pomdp.map_size[1], pomdp.map_size[2])
    P = Matrix(undef, pomdp.map_size[1], pomdp.map_size[2])
    for x in 1:pomdp.map_size[1]
        for y in 1:pomdp.map_size[2]
            pos = [x,y] # note flip
            input = network_input(b)
            input[1:2] = pos
            ŷ = f(input)
            v = ŷ[1]
            p = ŷ[2:end]
            V[x,y] = v
            # P[x,y] = p
            P[x,y] = mean(rand(SparseCat(1:length(p), p)) for _ in 1:1000)
        end
    end
    return V, P
end


function rocksample_belief_viz(pomdp::POMDP, b)
    plot(xlims=(1, pomdp.map_size[1]), ylims=(1, pomdp.map_size[2]), ratio=1)

    μ = mean(s->s.rocks, b.particles)

    for (i,rocks) in enumerate(pomdp.rocks_positions)
        scatter!([rocks[1]], [rocks[2]], label=false, c=:black, msc=:white, ms=3, widen=false)
    end
    plot!()
end