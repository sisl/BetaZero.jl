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
    action_size::Int # [REQUIRED] Number of actions in the action space
    input_size = (30,30,5) # Input belief size
    training_epochs::Int = 1000 # Number of network training updates
    n_samples::Int = 10_000 # Number of samples (i.e., simulated POMDP time steps from data collection) to use during training + validation
    normalize_input::Bool = true # Normalize input data to standard normal (0 mean)
    normalize_output::Bool = true # Normalize output (target) data to standard normal (0 mean)
    training_split::Float64 = 0.8 # Training / validation split (Default: 80/20)
    batchsize::Int = 512 # Batch size
    learning_rate::Float64 = 0.001 # Learning rate for ADAM optimizer during training
    λ_regularization::Float64 = 1e-5 # Parameter for L2-norm regularization
    optimizer = Adam # Training optimizer (e.g., Adam, Descent, Nesterov)
    loss_func::Function = Flux.Losses.mae # MAE works well for problems with large returns around zero, and spread out otherwise.
    activation::Function = relu # Activation function
    layer_size::Int = 64 # Number of connections in fully connected layers (for CNN, refers to fully connected "head" layers)
    use_cnn::Bool = false # Use convolutional neural network
    cnn_params::NamedTuple = (filter=(5,5), num_filters=[6, 16], num_dense=[120, 84])
    use_dropout::Bool = false # Indicate the use of dropout layers
    p_dropout::Float64 = 0.2 # Probability of dropout
    use_batchnorm::Bool = false # Indicate the use of batch normalization layers
    batchnorm_momentum = 0.1f0 # Momentum parameter for batch normalization
    device = gpu # Indicate what device to train on (`gpu` or `cpu`)
    use_checkpoint::Bool = true # Save networks along the way to use based on minimum validation loss
    checkpoint_frequency::Int = 1 # How often do we evaluate and save a checkpoint?
    checkpoint_validation_loss::Bool = true # Checkpoint based on minimum validation loss (`false` = checkpointing on training loss)
    verbose_update_frequency::Int = training_epochs # Frequency of printed training output
    verbose_plot_frequency::Number = Inf # Frequency of plotted training/validation output
end


"""
Parameters for Gaussian procces surrogate model.
"""
@with_kw mutable struct BetaZeroGPParameters
    kernel_v = Matern(1/2, 2.5, 2.5) # Kernel for value surrogate
    kernel_p = SE(1.0, 1.0) # Kernel for policy surrogate (deepcopied to have one GP for each action)
    input_size = (2,) # Input belief size (e.g., [μ, σ])
    n_samples::Int = 100 # Number of samples (i.e., simulated POMDP time steps from data collection) to use during training + validation
    training_split::Float64 = 0.8 # Training / validation split (Default: 80/20)
    λ_lcb::Float64 = 0.1 # Parameter for lower confidence bound (LCB)
    opt_iterations::Int = 10_000 # Number of iterations for GP optimization (different from `optimze`)
    tune_individual_action_gps::Bool = false # Indicate if policy GP should tune hyperparameters on each individual action GP (or share parameters across all actions)
    verbose_plot::Bool = false # Indicator of plotted training/validation output
    verbose_show_value_plot::Bool = false # Show value heatmap
    verbose_show_policy_plot::Bool = false # Show policy heatmap
    verbose_show_individual_action_plots::Bool = false # Show heatmap for individual action GPs
    use_lcb::Bool = false # Use LCB when predicting
    use_lcb_initial::Bool = false # Use LCB when predicting initial GP (generally keep at `false` as the uncertainty is high with mean zero, leading to large negative initial value estimates)
    optimize::Bool = true # Indicate if GP params should be optimized to the validation data
end


"""
Collection of parameters, useful for storing with the policy to indicate what parameters were used.
"""
mutable struct ParameterCollection
    params::BetaZeroParameters
    nn_params::BetaZeroNetworkParameters
    gp_params::BetaZeroGPParameters
end
