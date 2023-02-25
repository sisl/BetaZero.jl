"""
Parameters for the BetaZero algorithm.
"""
@with_kw mutable struct BetaZeroParameters
    n_iterations::Int = 1 # BetaZero policy iterations (primary outer loop).
    n_data_gen::Int = 10 # Number of episodes to run for training/validation data generation.
    n_evaluate::Int = 0 # Number of episodes to run for surrogate evaluation and comparison.
    n_holdout::Int = 10 # Number of episodes to run for a holdout test set (on a fixed, non-training or evaluation set).
    n_buffer::Int = n_iterations # Number of iterations to keep data for surrogate training (NOTE: each simulation has multiple time steps of data, not counted in this number. This number corresponds to the number of iterations, i.e., set to 2 if you want to keep data from the previous 2 policy iterations.)
    λ_ucb::Real = 0.0 # Upper confidence bound parameter: μ + λσ # TODO: Remove?
    use_nn::Bool = true # Use neural network as the surrogate model
end


"""
Parameters for neural network surrogate model.
"""
@with_kw mutable struct BetaZeroNetworkParameters
    input_size = (30,30,5)
    training_epochs::Int = 1000 # Number of network training updates
    n_samples::Int = 10_000 # Number of samples (i.e., simulated POMDP time steps from data collection) to use during training + validation
    normalize_target::Bool = true # Normalize target data to standard normal (0 mean)
    training_split::Float64 = 0.8 # Training / validation split (Default: 80/20)
    batchsize::Int = 512
    learning_rate::Float64 = 0.01 # Learning rate for ADAM optimizer during training
    λ_regularization::Float64 = 0.0001 # Parameter for L2-norm regularization
    loss_func::Function = Flux.Losses.mae # MAE works well for problems with large returns around zero, and spread out otherwise.
    device = gpu
    verbose_update_frequency::Int = training_epochs # Frequency of printed training output
    verbose_plot_frequency::Number = 10 # Frequency of plotted training/validation output
end


"""
Parameters for Gaussian procces surrogate model.
"""
@with_kw mutable struct BetaZeroGPParameters
    kernel = Matern(1/2, 1.0, 10.0)
    input_size = (6,)
    n_samples::Int = 500 # Number of samples (i.e., simulated POMDP time steps from data collection) to use during training + validation
    training_split::Float64 = 0.8 # Training / validation split (Default: 80/20)
    λ_lcb::Float64 = 0.1 # Parameter for lower confidence bound (LCB)
    verbose_plot::Bool = false # Indicator of plotted training/validation output
    use_lcb::Bool = true # Use LCB when predicting
    use_lcb_initial::Bool = false # Use LCB when predicting initial GP (generally keep at `false` as the uncertainty is high with mean zero, leading to large negative initial value estimates)
    optimize::Bool = false # Indicate if GP params should be optimized to the training data (generally keep at `false`)
end


"""
Collection of parameters, useful for storing with the policy to indicate what parameters were used.
"""
mutable struct ParameterCollection
    params::BetaZeroParameters
    nn_params::BetaZeroNetworkParameters
    gp_params::BetaZeroGPParameters
end
