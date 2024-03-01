# MineralExploration
This is the code for the intelligent prospector mineral exploration work conducted at Stanford University in a collaboration with SISL and SCERF. The source code provides a simulated mineral exploration problem and an implementation of the proposed sequential solution framework. The scripts needed to run the published experiments and additional trials are included in the `scripts` directory. All experiments were run in Julia 1.6.x. The specific versions of the required support packages are listed in the `project.toml`. 

## Installation 
The package can be installed using the standard Julia package manager utility `add` function. To add the package from a local source, simply run the command (from within the package manager environment) 

```
] add /PATH_TO_SOURCE
```
 alternatively, you may navigate to the home directory of the source and run 
 ```
 ] add . 
 ```
 If you would like to develop within the code base, we recommend building the source directly within a virtual environment. The requireed packages can be installed using the Julia `build` command from the source home directory. 

## Code Organization
All of the source code is in the `src` directory. The `MineralExploration.jl` file defines the main module strucutre. In this file, the source for all of the exposed structures and functions can be located. The `parameters` directory contains the data used to fit the variograms in the published experiments. The `scripts` directory contains the scripts to run the expeirments and tests. 

## Example Use
The `solve_pomcpow.jl` script provides the best example of how to use this codebase. Key snippets are provided here.
```
using POMDPs
...

using MineralExploration

# 1) Build problem model
mainbody = SingleFixedNode()
m = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=GRID_SPACING+1, grid_spacing=GRID_SPACING,
                            mainbody_gen=mainbody, max_movement=MAX_MOVEMENT)
                            # , geodist_type=GSLIBDistribution)
initialize_data!(m, N_INITIAL)
ds0 = POMDPs.initialstate_distribution(m)

# 2) Sample initial state
s0 = rand(ds0)

# 3) Create belief
up = MEBeliefUpdater(m, 1000, 2.0)
println("Initializing belief...")
b0 = POMDPs.initialize_belief(up, ds0)
println("Belief Initialized!")


ore_maps = [p.ore_map for p in b0.particles];

# 4) Create POMCPOW instance
next_action = NextActionSampler() 
solver = POMCPOWSolver(tree_queries=10000,
                       check_repeat_obs=true,
                       check_repeat_act=true,
                       next_action=next_action,
                       k_action=2.0,
                       alpha_action=0.25,
                       k_observation=2.0,
                       alpha_observation=0.1,
                       criterion=POMCPOW.MaxUCB(100.0),
                       final_criterion=POMCPOW.MaxQ(),
                       # final_criterion=POMCPOW.MaxTries(),
                       estimate_value=0.0
                       # estimate_value=leaf_estimation
                       )
planner = POMDPs.solve(solver, m)

# 5) Run simulation
b_new = nothing
a_new = nothing
discounted_return = 0.0
B = [b0]
AE = Float64[]
ME = Float64[]
println("Entering Simulation...")
for (sp, a, r, bp, t) in stepthrough(m, planner, up, b0, s0, "sp,a,r,bp,t", max_steps=50)
    ...
    
    @show t
    @show a.type
    @show a.coords
    @show r
    @show sp.stopped
    @show bp.stopped
    volumes = [sum(p.ore_map .>= m.massive_threshold) for p in bp.particles]
    mean_volume = mean(volumes)
    std_volume = std(volumes)
    volume_lcb = mean_volume - 1.0*std_volume
    push!(B, bp)
    @show mean_volume
    @show std_volume
    @show volume_lcb

    ...
    
    discounted_return += POMDPs.discount(m)^(t - 1)*r
end
```
In the above code, `...` denotes code from the original script that was omitted for brevity here. 

Note: To save data, the defalut location is a `data` folder in the source code home directory. To change this, change the file path strings referenced in the script. 
