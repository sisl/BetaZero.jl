"""
Parameters for the BetaZero algorithm.
"""
@with_kw mutable struct BetaZeroParameters
    n_iterations::Int = 10 # BetaZero policy iterations (primary outer loop).
    n_data_gen::Int = 100 # Number of episodes to run for training/validation data generation.
    n_evaluate::Int = 0 # Number of episodes to run for surrogate evaluation and comparison (when `n_evaluate == 0`, skip the evaluation step)
    n_holdout::Int = 0 # Number of episodes to run for a holdout test set (on a fixed, non-training or evaluation set).
    n_buffer::Int = 1 # Number of iterations to keep data for surrogate training (NOTE: each simulation has multiple time steps of data, not counted in this number. This number corresponds to the number of iterations, i.e., set to 2 if you want to keep data from the previous 2 policy iterations.)
    max_steps::Int = 100 # Maximum number of steps for each simulation.
    λ_ucb::Real = 0.0 # Upper confidence bound parameter for network evaluation/comparison: μ + λσ
    use_nn::Bool = true # Use neural network as the surrogate model
    use_q_weighted_counts::Bool = true # When collecting the policy data, weight the visit count distribution by the Q-values
    use_completed_policy_gumbel::Bool = false # When using the Gumbel solver, use the completed policy estimate as the policy data
    use_raw_policy_network::Bool = false # Generate data only from the raw policy network
    use_raw_value_network::Bool = false # Generate data only from the raw value network (given a `n_obs` below)
    raw_value_network_n_obs::Int = 1 # When using the raw value network via `use_raw_value_network`, specify number of observations per action to expand on
    skip_missing_reward_signal::Bool = false # When running MCTS episodes, filter out trajectories that had no reward signal (i.e., zero reward everywhere)
    train_missing_on_predicted::Bool = false # Use predicted value in place of missing reward signal episodes
    eval_on_accuracy::Bool = false # If evaluating (i.e., `n_evaluate > 0`), then base comparison on accuracy of the two networks
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
    sample_more_than_collected::Bool = true # Sample more data (with replacement) than is in the buffer
    batchsize::Int = 512 # Batch size
    learning_rate::Float64 = 0.001 # Learning rate for ADAM optimizer during training
    λ_regularization::Float64 = 1e-5 # Parameter for L2-norm regularization
    optimizer = Adam # Training optimizer (e.g., Adam, Descent, Nesterov)
    loss_func::Function = Flux.Losses.mse # MAE works well for problems with large returns around zero, and spread out otherwise.
    activation::Function = relu # Activation function
    layer_size::Int = 64 # Number of connections in fully connected layers (for CNN, refers to fully connected "head" layers)
    use_cnn::Bool = false # Use convolutional neural network
    use_deepmind_arch::Bool = false # Use simplified non-resnet architecture from AlphaZero
    cnn_params::NamedTuple = (filter=(5,5), num_filters=[64, 128], num_dense=[256, 256])
    use_dropout::Bool = false # Indicate the use of dropout layers
    p_dropout::Float64 = 0.2 # Probability of dropout
    use_batchnorm::Bool = false # Indicate the use of batch normalization layers
    batchnorm_momentum = 0.1f0 # Momentum parameter for batch normalization
    use_dirichlet_exploration::Bool = false # Apply Dirichlet noise to policy vector for exploration
    α_dirichlet::Float64 = 0.03 # Alpha parameter of the Dirichlet action noise distribution
    ϵ_dirichlet::Float64 = 0.25 # Weighting parameter for applying Dirichlet action noise
    zero_out_tried_actions::Bool = false # When selecting a next action to widen on, zero out the probabilities for already tried actions.
    next_action_return_argmax::Bool = false # Instead of sampling, return the argmax action during action widening
    use_epsilon_greedy::Bool = false # Use epsilon-greedy exploration during action widening
    ϵ_greedy::Float64 = 0.0 # Epsilon parameter to select random action during widening with probability ϵ_greedy
    classification_loss_weight::Float32 = 0.5f0 # Weight applied to the classification (policy) component of the loss function
    use_lk_loss::Bool = false # Use KL-divergence as classification (policy) loss (for Gumbel solver)
    incremental_save::Bool = false # Incrementally save off policy every iteration (TODO: fix undefined reference error)
    policy_filename::String = "betazero_policy.bson" # Filename when incrementally saving off poliy
    device = gpu # Indicate what device to train on (`gpu` or `cpu`)
    use_checkpoint::Bool = true # Save networks along the way to use based on minimum validation loss
    checkpoint_frequency::Int = 1 # How often do we evaluate and save a checkpoint?
    checkpoint_validation_loss::Bool = true # Checkpoint based on minimum validation loss (`false` = checkpointing on training loss)
    verbose_update_frequency::Int = training_epochs # Frequency of printed training output
    verbose_plot_frequency::Number = Inf # Frequency of plotted training/validation output
    display_plots::Bool = false # Display training and validation plots after training
    save_plots::Bool = false # Save training and validation plots after training
    plot_curve_filename::String = "training_curve.png" # Filename for the training/validation loss curve
    plot_value_distribution_filename::String = "value_distribution.png" # Filename for the distribution of values (model vs. data)
    plot_training_bias_filename::String = "training_data.png" # Filename the bias plots for training model vs. data
    plot_validation_bias_filename::String = "validation_data.png" # Filename the bias plots for validation model vs. data
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
