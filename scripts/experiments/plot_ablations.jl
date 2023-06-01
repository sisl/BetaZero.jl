ENV["LAUNCH_PARALLEL"] = false
!@isdefined(LightDark) && include("../launch_lightdark.jl")

#—————————————————————————————————————#
!@isdefined(QN_SOLVERS) && include("load_ablation_q_weighting.jl")
plot_ablations(QN_SOLVERS; use_pgf=true, max_iterations=20, xticks=2000:2000:10000, ablation_type=:q_weighting)

#—————————————————————————————————————#
!@isdefined(PA_SOLVERS) && include("load_ablation_action_selection.jl")
plot_ablations(PA_SOLVERS; use_pgf=true, ablation_type=:action_selection)

#—————————————————————————————————————#
!@isdefined(BR_SOLVERS) && include("load_ablation_belief_representation.jl")
plot_ablations(BR_SOLVERS; use_pgf=true, max_iterations=30, xticks=5000:5000:15000, ablation_type=:belief_rep)

nothing # REPL