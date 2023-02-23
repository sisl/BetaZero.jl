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
    kernel_params = (1/2, 10.0, 10.0) # Parameters for the chosen kernel (e.g., (ν, ll, lσ) for the Matern kernel)
    kernel = Matern(kernel_params...)
    input_size = (6,)
    n_samples::Int = 500 # Number of samples (i.e., simulated POMDP time steps from data collection) to use during training + validation
    training_split::Float64 = 0.8 # Training / validation split (Default: 80/20)
    λ_lcb::Float64 = 0.1 # Parameter for lower confidence bound (LCB)
    verbose_plot::Bool = false # Indicator of plotted training/validation output
    use_lcb::Bool = true # Use LCB when predicting
    use_lcb_initial::Bool = false # Use LCB when predicting initial GP (generally keep at `false` as the uncertainty is high with mean zero, leading to large negative initial value estimates)
    optimize::Bool = false # Indicate if GP params should be optimized to the training data (generally keep at `false`)
end