# BetaZero.jl

Belief-state planning algorithm for POMDPs using learned approximations; integrated into the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) ecosystem.

<!-- ![light mode](/media/betazero.svg#gh-light-mode-only) -->
<!-- ![dark mode](/media/betazero-dark.svg#gh-dark-mode-only) -->
<img src="./media/betazero.svg">


## Installation

To install the BetaZero solver, run:

```bash
julia install.jl
```

This also installs the [MCTS.jl](https://github.com/JuliaPOMDP/MCTS.jl) PUCT fork, the `RemoteJobs` package, and the `ParticleBeliefs` wrapper.

(**Optional**) To install the supporting example POMDP models (e.g., LightDark), run:
```bash
julia install.jl --models
```


## Usage

The following code sets up the necessary interface functions `BetaZero.input_representation` and the optional `BetaZero.accuracy` for the _LightDark_ POMDP problem and solves it using BetaZero.

```julia
using BetaZero
using LightDark

pomdp = LightDarkPOMDP()
up = BootstrapFilter(pomdp, 500)

function BetaZero.input_representation(b::ParticleCollection{LightDarkState})
    # Function to get belief representation as input to neural network.
    μ, σ = mean_and_std(s.y for s in particles(b))
    return Float32[μ, σ]
end

function BetaZero.accuracy(pomdp::LightDarkPOMDP, b0, s0, states, actions, returns)
    # Function to determine accuracy of agent's final decision.
    return returns[end] == pomdp.correct_r
end

solver = BetaZeroSolver(pomdp=pomdp,
                        updater=up,
                        params=BetaZeroParameters(
                            n_iterations=50,
                            n_data_gen=50,
                        ),
                        nn_params=BetaZeroNetworkParameters(
                            pomdp, up;
                            training_epochs=50,
                            n_samples=100_000,
                            batchsize=1024,
                            learning_rate=1e-4,
                            λ_regularization=1e-5,
                            use_dropout=true,
                            p_dropout=0.2,
                        ),
                        verbose=true,
                        collect_metrics=true,
                        plot_incremental_data_gen=true)

policy = solve(solver, pomdp)
save_policy(policy, "policy.bson")
save_solver(solver, "solver.bson")
```
This example is also located at: [`scripts/readme_example.jl`](https://github.com/sisl/BetaZero.jl/blob/main/scripts/readme_example.jl)


### Parallel Usage

Using the provided light-weight `RemoteJobs` package, you can easily launch workers to run the MCTS data collection in parallel.

```julia
using RemoteJobs
machine_specs = [("user@host1", 25), ("user@host2", 25)] # launch 50 processes across two separate hosts
launch_remote_workers(machine_specs)

@everywhere begin
    using BetaZero
    using LightDark

    pomdp = LightDarkPOMDP()
    up = BootstrapFilter(pomdp, 500)

    function BetaZero.input_representation(b::ParticleCollection{LightDarkState})
        # Function to get belief representation as input to neural network.
        μ, σ = mean_and_std(s.y for s in particles(b))
        return Float32[μ, σ]
    end

    function BetaZero.accuracy(pomdp::LightDarkPOMDP, b0, s0, states, actions, returns)
        # Function to determine accuracy of agent's final decision.
        return returns[end] == pomdp.correct_r
    end
end

solver = BetaZeroSolver(pomdp=pomdp,
                        updater=up,
                        params=BetaZeroParameters(
                            n_iterations=50,
                            n_data_gen=500, # Note increased to 500 when running in parallel.
                        ),
                        nn_params=BetaZeroNetworkParameters(
                            pomdp, up;
                            training_epochs=50,
                            n_samples=100_000,
                            batchsize=1024,
                            learning_rate=1e-4,
                            λ_regularization=1e-5,
                            use_dropout=true,
                            p_dropout=0.2,
                        ),
                        verbose=true,
                        collect_metrics=true,
                        plot_incremental_data_gen=true)

policy = solve(solver, pomdp)
save_policy(policy, "policy.bson")
save_solver(solver, "solver.bson")
```
This example is also located at: [`scripts/readme_example_parallel.jl`](https://github.com/sisl/BetaZero.jl/blob/main/scripts/readme_example_parallel.jl)



### Other Examples

See the following files for more examples:

- LightDark POMDP
    - Training [`scripts/train_lightdark.jl`](https://github.com/sisl/BetaZero.jl/blob/main/scripts/train_lightdark.jl)
    - Solver [`scripts/solver_lightdark.jl`](https://github.com/sisl/BetaZero.jl/blob/main/scripts/solver_lightdark.jl)
    - Representation [`scripts/representation_lightdark.jl`](https://github.com/sisl/BetaZero.jl/blob/main/scripts/representation_lightdark.jl)
- RockSample POMDP
    - Training [`scripts/train_rocksample.jl`](https://github.com/sisl/BetaZero.jl/blob/main/scripts/train_rocksample.jl)
    - Solver [`scripts/solver_rocksample.jl`](https://github.com/sisl/BetaZero.jl/blob/main/scripts/solver_rocksample.jl)
    - Representation [`scripts/representation_rocksample.jl`](https://github.com/sisl/BetaZero.jl/blob/main/scripts/representation_rocksample.jl)
- Mineral Exploration POMDP
    - Training [`scripts/train_minex.jl`](https://github.com/sisl/BetaZero.jl/blob/main/scripts/train_minex.jl)
    - Solver [`scripts/solver_minex.jl`](https://github.com/sisl/BetaZero.jl/blob/main/scripts/solver_minex.jl)
    - Representation [`scripts/representation_minex.jl`](https://github.com/sisl/BetaZero.jl/blob/main/scripts/representation_minex.jl)


## Parameters
See the parameters, their descriptions, and their defaults in [`src/parameters.jl`](https://github.com/sisl/BetaZero.jl/blob/main/src/parameters.jl)


## Directory structure

    .
    ├── media                   # Image files.
    ├── scripts                 # Example scripts for training POMDPs, visualizations, and baselines.
    ├── src                     # Core Julia package files with BetaZero implementation and supporting code.
    ├── submodules              # Submodules used by the scripts.
    │   └── CryingBaby          # POMDP model for the crying baby problem.
    │   └── LightDark           # POMDP model for the light dark problem.
    │   └── MCTS                # MCTS.jl [1] fork with PUCT and BetaZero root action criteria.
    │   └── MinEx               # POMDP model for the mineral exploration problem.
    │   └── ParticleBeliefs     # Lightweight wrapper for a particle filter that stores actions and observations.
    │   └── RemoteJobs          # Lightweight package for launching remote workers and syncing code.
    │   └── RobotLocalization   # POMDP model for the robot localization problem (adapted from RoombaPOMDPs.jl [2]).
    │   └── Tiger               # POMDP model for the tiger problem.
    └── tex                     # LaTeX files for the TikZ mineral exploration policy maps.

[1] https://github.com/JuliaPOMDP/MCTS.jl

[2] https://github.com/sisl/RoombaPOMDPs.jl

## Citation

> Under review.