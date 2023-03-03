using BetaZero
using LightDark
using ParticleFilters
using Plots; default(fontfamily="Computer Modern", framestyle=:box)
using POMDPs
using POMDPTools
using Random
using StatsBase

# seed=3348096912 (reversal)
function plot_lightdark(pomdp::LightDarkPOMDP, policy::Policy, up::Updater; max_steps=20, seed=rand(1:typemax(UInt32)))
    Random.seed!(seed)
    @info "seed = $seed"

    ds0 = initialstate(pomdp)
    b0 = initialize_belief(up, ds0)
    s0 = rand(b0)
    S = [s0]
    A = [0.0]
    O = [0.0]
    B = [b0]
    R = [0.0]
    rd = x->round(x, digits=4)
    for (t,s,a,o,b,r,sp,bp) in stepthrough(pomdp, policy, up, b0, s0, "t,s,a,o,b,r,sp,bp"; max_steps)
        ỹ, σ = mean_and_std(s.y for s in ParticleFilters.particles(b))
        push!(S, s)
        push!(A, a)
        push!(O, o)
        push!(R, r)
        push!(B, bp)
        @info "[$t] \t s = $(rd(s.y))\t a = $(rd(a))\t r = $(rd(r))\t b = $(rd.([ỹ, σ]))"
    end

    if hasproperty(policy, :surrogate)
        G = BetaZero.compute_returns(R; γ=POMDPs.discount(pomdp))
        Ṽ = [Float64(BetaZero.value_lookup(b, policy.surrogate)) for b in B]
        value_mae = mean(abs.(G .- Ṽ))
        @info "Value estimate MAE: $(round(value_mae, digits=4))"
    end

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

    # using ColorSchemes
    Y = map(s->s.y, S)
    Ỹ = map(b->mean(p.y for p in ParticleFilters.particles(b)), B)
    ymax = max(max_steps, max(maximum(Y), abs(minimum(Y))))*1.5
    xmax = max(length(S), max_steps)
    plt_lightdark = plot(xlims=(1, xmax), ylims=(-ymax, ymax), size=(700,300), margin=7Plots.mm, left_margin=20Plots.mm, legend=:outertop, xlabel="time", ylabel="state")
    heatmap!(1:xmax, range(-ymax, ymax, length=100), (x,y)->sqrt(std(observation(pomdp, LightDarkState(0, y)))), c=:grayC)
    hline!([0], c=:green, style=:dash, label="goal")
    plot_beliefs(B; hold=true)
    plot!(eachindex(S), Y, c=:red, lw=2, label="trajectory", alpha=0.5)
    plot!(eachindex(S), Ỹ, c=:blue, lw=1, ls=:dash, label="believed traj.", alpha=0.5)
    scatter!(eachindex(S), O, ms=2, c=:cyan, msc=:black, label="observation")
    return plt_lightdark
end
