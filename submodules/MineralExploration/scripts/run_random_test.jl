using Revise

using DelimitedFiles
using POMDPs
using POMDPPolicies
using POMCPOW
using Plots
using Statistics
using StatsBase

using MineralExploration

function run_test(SAVE_DIR, CASE_DIR, MAX_MOVEMENT, MAINBODY_STR, display_figs=false)
    N = 100
    N_INITIAL = 0
    MAX_BORES = 10
    GRID_SPACING = 1
    # MAX_MOVEMENT = 0
    # CASE_DIR = "./data/single_fixed_test_cases.jld"
    # SAVE_DIR = "./data/tests/random/single_fixed_unconstrained/"

    # SAVE_DIR = parsed_args["save_dir"]
    # CASE_DIR = parsed_args["case_dir"]
    # MAX_MOVEMENT = parsed_args["max_movement"]
    # MAINBODY_STR = parsed_args["mainbody"]

    if MAINBODY_STR == "single_fixed"
        mainbody = SingleFixedNode()
    elseif MAINBODY_STR == "single_variable"
        mainbody = SingleVarNode()
    elseif MAINBODY_STR == "two"
        mainbody = MultiVarNode()
    end

    m = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=GRID_SPACING+1, grid_spacing=GRID_SPACING,
                                mainbody_gen=mainbody, max_movement=MAX_MOVEMENT)
    initialize_data!(m, N_INITIAL)
    ds0 = POMDPs.initialstate_distribution(m)

    s0s = nothing
    if isfile(CASE_DIR)
        s0s = MineralExploration.load(CASE_DIR)["states"]
    else
        println("Saved cases not found! Generating new cases...")
        s0s = gen_cases(ds0, N, CASE_DIR)
    end

    up = MEBeliefUpdater(m, 1000, 2.0)
    b0 = POMDPs.initialize_belief(up, ds0)

    next_action = NextActionSampler()
    planner = POMDPPolicies.RandomPolicy(m)

    returns = Float64[]
    ores = Float64[]
    decisions = Symbol[]
    distances = Vector{Float64}[]
    abs_errs = Vector{Float64}[]
    vol_stds = Vector{Float64}[]
    n_drills = Int64[]
    for i = 1:N
        println("Running Trial $i")
        s0 = s0s[i]
        results = run_trial(m, up, planner, s0, b0, save_dir=nothing, display_figs=false, verbose=false)
        push!(returns, results[1])
        push!(ores, results[6])
        push!(decisions, results[7])
        push!(distances, results[2])
        push!(abs_errs, results[3])
        push!(vol_stds, results[4])
        push!(n_drills, results[5])
        true_ore = s0.ore_map
        massive_ore = s0.ore_map .>= m.massive_threshold
        noise_ore = s0.ore_map - s0.mainbody_map
        path = string(SAVE_DIR, "true_ore.txt")
        open(path, "a") do io
            writedlm(io, reshape(true_ore, :, 1))
        end
        path = string(SAVE_DIR, "massive_ore.txt")
        open(path, "a") do io
            writedlm(io, reshape(massive_ore, :, 1))
        end
        path = string(SAVE_DIR, "noise_ore.txt")
        open(path, "a") do io
            writedlm(io, reshape(noise_ore, :, 1))
        end
    end

    fig, mu, sig = plot_history(distances, 10, "Distance to Center", "Distance")
    path = string(SAVE_DIR, "distances.png")
    savefig(fig, path)
    path = string(SAVE_DIR, "distances.pdf")
    savefig(fig, path)
    if display_figs
        display(fig)
    end

    fig, mu, sig = plot_history(abs_errs, 10, "Mean Absolute Errors", "MAE")
    path = string(SAVE_DIR, "mae.png")
    savefig(fig, path)
    path = string(SAVE_DIR, "mae.pdf")
    savefig(fig, path)
    if display_figs
        display(fig)
    end


    for vol_std in vol_stds
        vol_std ./= vol_std[1]
    end
    fig, mu, sig = plot_history(vol_stds, 10, "Volume Standard Deviation Ratio", "σ/σ₀")
    path = string(SAVE_DIR, "vol_stds.png")
    savefig(fig, path)
    path = string(SAVE_DIR, "vol_stds.pdf")
    savefig(fig, path)
    if display_figs
        display(fig)
    end
    h = fit(Histogram, n_drills)
    h = StatsBase.normalize(h, mode=:probability)
    b_hist = plot(h, title="Number of Bores", legend=:none)
    path = string(SAVE_DIR, "bore_hist.png")
    savefig(b_hist, path)
    path = string(SAVE_DIR, "bore_hist.pdf")
    savefig(b_hist, path)
    if display_figs
        display(b_hist)
    end

    abandoned = [a == :abandon for a in decisions]
    mined = [a == :mine for a in decisions]

    profitable = ores .> m.extraction_cost
    lossy = ores .<= m.extraction_cost

    n_profitable = sum(profitable)
    n_lossy = sum(lossy)

    profitable_mined = sum(mined.*profitable)
    profitable_abandoned = sum(abandoned.*profitable)

    lossy_mined = sum(mined.*lossy)
    lossy_abandoned = sum(abandoned.*lossy)

    mined_profit = sum(mined.*(ores .- m.extraction_cost))
    available_profit = sum(profitable.*(ores .- m.extraction_cost))
    # println("Mean Bores: $mean_drills, Mined Bores: $mined_drills, Abandon Bores: $abandoned_drills")

    h = fit(Histogram, ores[mined] .- m.extraction_cost, [-20:10:100;])
    # h = StatsBase.normalize(h, mode=:probability)
    ore_hist = plot(h, title="Mined Profit", legend=:none)
    path = string(SAVE_DIR, "ore_hist.png")
    savefig(ore_hist, path)
    path = string(SAVE_DIR, "ore_hist.pdf")
    savefig(ore_hist, path)
    if display_figs
        display(ore_hist)
    end

    mean_drills = mean(n_drills)
    mined_drills = sum(n_drills.*mined)/sum(mined)
    abandoned_drills = sum(n_drills.*abandoned)/sum(abandoned)

    path = string(SAVE_DIR, "performance_summary.txt")
    open(path, "w") do io
        write(io, "Available Profit: $available_profit, Mined Profit: $mined_profit")
        println("Available Profit: $available_profit, Mined Profit: $mined_profit")
        write(io, "\nProfitable: $(sum(profitable)), Mined: $profitable_mined, Abandoned: $profitable_abandoned")
        println("Profitable: $(sum(profitable)), Mined: $profitable_mined, Abandoned: $profitable_abandoned")
        write(io, "\nLossy: $(sum(lossy)), Mined: $lossy_mined, Abandoned: $lossy_abandoned")
        println("Lossy: $(sum(lossy)), Mined: $lossy_mined, Abandoned: $lossy_abandoned")
        write(io, "\nMean Bores: $mean_drills, Mined Bores: $mined_drills, Abandon Bores: $abandoned_drills")
        println("Mean Bores: $mean_drills, Mined Bores: $mined_drills, Abandon Bores: $abandoned_drills")
    end
end
