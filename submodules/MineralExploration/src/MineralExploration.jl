module MineralExploration

using BeliefUpdaters
using CSV
using DataFrames
using DelimitedFiles
using Distributions
using GeoStats
using JLD
using LinearAlgebra
using Parameters
using Plots
using POMCPOW
using POMDPModelTools
using POMDPSimulators
using POMDPs
using Random
using StatsBase
using StatsPlots
using Statistics


export
        MEState,
        MEObservation,
        MEAction,
        RockObservations,
        GeoDist
include("common.jl")

export
        GeoStatsDistribution,
        kriging
include("geostats.jl")

export
        GSLIBDistribution,
        kriging
include("gslib.jl")

export
        MainbodyGen,
        SingleFixedNode,
        SingleVarNode,
        MultiVarNode
include("mainbody.jl")

export
        MineralExplorationPOMDP,
        MEInitStateDist,
        initialize_data!
include("pomdp.jl")

export
        MEBelief,
        MEBeliefUpdater
include("beliefs.jl")

export
        NextActionSampler,
        ExpertPolicy,
        RandomSolver,
        GridPolicy,
        leaf_estimation
include("solver.jl")

export
        GPNextAction
include("action_selection.jl")

export
        plot_history,
        run_trial,
        gen_cases
include("utils.jl")

end
