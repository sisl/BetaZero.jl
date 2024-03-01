function plot_history(hs::Vector, n_max::Int64=10,
                        title::Union{Nothing, String}=nothing,
                        y_label::Union{Nothing, String}=nothing,
                        box_plot::Bool=false)
    μ = Float64[]
    σ = Float64[]
    vals_vector = Vector{Float64}[]
    for i = 1:n_max
        vals = Float64[]
        for h in hs
            if length(h) >= i
                push!(vals, h[i])
            end
        end
        push!(μ, mean(vals))
        push!(σ, std(vals)/sqrt(length(vals)))
        push!(vals_vector, vals)
    end
    σ .*= 1.0 .- isnan.(σ)
    if isa(title, String)
        if box_plot
            fig = plot(μ, legend=:none, title=title, ylabel=y_label)
            for (i, vals) in enumerate(vals_vector)
                boxplot!(fig, repeat([i], outer=length(vals)), vals, color=:white, outliers=false)
            end
        else
            fig = plot(μ, yerror=σ, legend=:none, title=title, ylabel=y_label)
        end
    else
        if box_plot
            fig = plot(μ, legend=:none)
            for (i, vals) in enumerate(vals_vector)
                boxplot!(fig, i, vals, color=:white, outliers=false)
            end
        else
            fig = plot(μ, yerror=σ, legend=:none)
        end
    end
    return (fig, μ, σ)
end

function gen_cases(ds0::MEInitStateDist, n::Int64, save_dir::Union{String, Nothing}=nothing)
    states = MEState[]
    for i = 1:n
        push!(states, rand(ds0))
    end
    if isa(save_dir, String)
        save(save_dir, "states", states)
    end
    return states
end

function run_trial(m::MineralExplorationPOMDP, up::POMDPs.Updater,
                policy::POMDPs.Policy, s0::MEState, b0::MEBelief;
                display_figs::Bool=true, save_dir::Union{Nothing, String}=nothing,
                verbose::Bool=true)
    if verbose
        println("Initializing belief...")
    end
    if verbose
        println("Belief Initialized!")
    end

    ore_fig = heatmap(s0.ore_map[:,:,1], title="True Ore Field", fill=true, clims=(0.0, 1.0))
    s_massive = s0.ore_map .>= m.massive_threshold
    r_massive = sum(s_massive)
    mass_fig = heatmap(s_massive[:,:,1], title="Massive Ore Deposits: $r_massive", fill=true, clims=(0.0, 1.0))
    b0_fig = plot(b0)

    vols = [sum(p.ore_map .>= m.massive_threshold) for p in b0.particles]
    mean_vols = round(mean(vols), digits=2)
    std_vols = round(std(vols), digits=2)
    if verbose
        println("Vols: $mean_vols ± $std_vols")
    end
    h = fit(Histogram, vols, [0:10:300;])
    h = normalize(h, mode=:probability)

    b0_hist = plot(h, title="Belief Volumes t=0\nμ=$mean_vols, σ=$std_vols", legend=:none)
    plot!(b0_hist, [r_massive, r_massive], [0.0, maximum(h.weights)], linecolor=:red, linewidth=3)
    if isa(save_dir, String)
        path = string(save_dir, "ore_map.png")
        savefig(ore_fig, path)

        path = string(save_dir, "mass_map.png")
        savefig(mass_fig, path)

        path = string(save_dir, "b0.png")
        savefig(b0_fig, path)

        path = string(save_dir, "b0_hist.png")
        savefig(b0_hist, path)
    end
    if display_figs
        display(ore_fig)
        display(mass_fig)
        display(b0_fig)
        display(b0_hist)
    end
    b_mean, b_std = MineralExploration.summarize(b0)
    path = string(save_dir, "belief_mean.txt")
    open(path, "w") do io
        writedlm(io, reshape(b_mean, :, 1))
    end
    path = string(save_dir, "belief_std.txt")
    open(path, "w") do io
        writedlm(io, reshape(b_std, :, 1))
    end

    last_action = :drill
    n_drills = 0
    discounted_return = 0.0
    ae = mean(abs.(vols .- r_massive))
    abs_errs = Float64[ae]
    vol_stds = Float64[std_vols]
    dists = Float64[]
    if verbose
        println("Entering Simulation...")
    end
    for (sp, a, r, bp, t) in stepthrough(m, policy, up, b0, s0, "sp,a,r,bp,t", max_steps=50)
        discounted_return += POMDPs.discount(m)^(t - 1)*r
        dist = sqrt(sum(([a.coords[1], a.coords[2]] .- 25.0).^2)) #TODO only for single fixed
        last_action = a.type

        b_fig = plot(bp, t)
        vols = [sum(p.ore_map .>= m.massive_threshold) for p in bp.particles]
        mean_vols = round(mean(vols), digits=2)
        std_vols = round(std(vols), digits=2)
        if verbose
            @show t
            @show a.type
            @show a.coords
            println("Vols: $mean_vols ± $std_vols")
        end

        if a.type == :drill
            n_drills += 1
            ae = mean(abs.(vols .- r_massive))
            push!(dists, dist)
            push!(abs_errs, ae)
            push!(vol_stds, std_vols)

            h = fit(Histogram, vols, [0:10:300;])
            h = normalize(h, mode=:probability)
            b_hist = plot(h, title="Belief Volumes t=$t\nμ=$mean_vols, σ=$std_vols", legend=:none)
            plot!(b_hist, [r_massive, r_massive], [0.0, maximum(h.weights)], linecolor=:red, linewidth=3)
            if isa(save_dir, String)
                path = string(save_dir, "b$t.png")
                savefig(b_fig, path)

                path = string(save_dir, "b$(t)_hist.png")
                savefig(b_hist, path)
            end
            if display_figs
                display(b_fig)
                display(b_hist)
            end
            b_mean, b_std = MineralExploration.summarize(bp)
            path = string(save_dir, "belief_mean.txt")
            open(path, "a") do io
                writedlm(io, reshape(b_mean, :, 1))
            end
            path = string(save_dir, "belief_std.txt")
            open(path, "a") do io
                writedlm(io, reshape(b_std, :, 1))
            end
        end
    end
    if verbose
        println("Discounted Return: $discounted_return")
    end
    ts = [1:length(abs_errs);] .- 1
    dist_fig = plot(ts[2:end], dists, title="Bore Distance to Center",
                    xlabel="Time Step", ylabel="Distance", legend=:none)
    abs_err_fig = plot(ts, abs_errs, title="Absolute Volume Error",
                    xlabel="Time Step", ylabel="Absolute Error", legend=:none)
    vols_fig = plot(ts, vol_stds./vol_stds[1], title="Volume Standard Deviation",
                    xlabel="Time Step", ylabel="Standard Deviation", legend=:none)
    if isa(save_dir, String)
        path = string(save_dir, "dists.png")
        savefig(dist_fig, path)

        path = string(save_dir, "abs_err.png")
        savefig(abs_err_fig, path)

        path = string(save_dir, "vol_std.png")
        savefig(vols_fig, path)
    end
    if display_figs
        display(dist_fig)
        display(abs_err_fig)
        display(vols_fig)
    end
    return (discounted_return, dists, abs_errs, vol_stds, n_drills, r_massive, last_action)
end
