abstract type MainbodyGen end

## Single Fixed Node
@with_kw struct SingleFixedNode <: MainbodyGen
    grid_dims::Tuple{Int64, Int64, Int64} = (50, 50, 1)
    mainbody_loc::Vector{Float64} = [25.0, 25.0]
    mainbody_var_min::Float64 = 50.0
    mainbody_var_max::Float64 = 80.0
end

function Base.rand(rng::Random.AbstractRNG, mb::SingleFixedNode)
    x_dim = mb.grid_dims[1]
    y_dim = mb.grid_dims[2]
    lode_map = zeros(Float64, x_dim, y_dim)
    mainbody_var = rand(rng)*(mb.mainbody_var_max - mb.mainbody_var_min) + mb.mainbody_var_min
    cov = Distributions.PDiagMat([mainbody_var, mainbody_var])
    mvnorm = MvNormal(mb.mainbody_loc, cov)
    for i = 1:x_dim
        for j = 1:y_dim
            lode_map[i, j] = pdf(mvnorm, [float(i), float(j)])
        end
    end
    return (lode_map, mainbody_var)
end

Base.rand(mb::SingleFixedNode) = rand(Random.GLOBAL_RNG, mb)

function perturb_sample(mb::SingleFixedNode, mainbody_var::Float64, noise::Float64)
    mainbody_var += 2.0*(rand() - 0.5)*noise
    mainbody_var = clamp(mainbody_var, 0.0, Inf)
    mainbody_map = zeros(Float64, Int(mb.grid_dims[1]), Int(mb.grid_dims[2]))
    cov = Distributions.PDiagMat([mainbody_var, mainbody_var])
    mvnorm = MvNormal(mb.mainbody_loc, cov)
    for i = 1:mb.grid_dims[1]
        for j = 1:mb.grid_dims[2]
            mainbody_map[i, j] = pdf(mvnorm, [float(i), float(j)])
        end
    end
    return (mainbody_map, mainbody_var)
end

## Single Variable Node
@with_kw struct SingleVarNode <: MainbodyGen
    grid_dims::Tuple{Int64, Int64, Int64} = (50, 50, 1)
    mainbody_loc_bounds::Vector{Float64} = [20.0, 30.0]
    mainbody_var_min::Float64 = 40.0
    mainbody_var_max::Float64 = 80.0
end

function Base.rand(rng::Random.AbstractRNG, mb::SingleVarNode)
    x_dim = mb.grid_dims[1]
    y_dim = mb.grid_dims[2]
    lode_map = zeros(Float64, x_dim, y_dim)
    mainbody_var = rand(rng)*(mb.mainbody_var_max - mb.mainbody_var_min) + mb.mainbody_var_min
    cov = Distributions.PDiagMat([mainbody_var, mainbody_var])
    mainbody_loc = rand(2).*(mb.mainbody_loc_bounds[2] - mb.mainbody_loc_bounds[1]) .+ mb.mainbody_loc_bounds[1]
    mvnorm = MvNormal(mainbody_loc, cov)
    for i = 1:x_dim
        for j = 1:y_dim
            lode_map[i, j] = pdf(mvnorm, [float(i), float(j)])
        end
    end
    mainbody_params = vcat(mainbody_loc, mainbody_var)
    return (lode_map, mainbody_params)
end

Base.rand(mb::SingleVarNode) = rand(Random.GLOBAL_RNG, mb)

function perturb_sample(mb::SingleVarNode, mainbody_params::Vector{Float64}, noise::Float64)
    mainbody_loc = mainbody_params[1:2]
    mainbody_var = mainbody_params[3]

    mainbody_loc += 2.0.*(rand(2) .- 0.5).*noise
    mainbody_var += 2.0*(rand() - 0.5)*noise
    mainbody_var = clamp(mainbody_var, 0.0, Inf)
    mainbody_map = zeros(Float64, Int(mb.grid_dims[1]), Int(mb.grid_dims[2]))
    cov = Distributions.PDiagMat([mainbody_var, mainbody_var])
    mvnorm = MvNormal(mainbody_loc, cov)
    for i = 1:mb.grid_dims[1]
        for j = 1:mb.grid_dims[2]
            mainbody_map[i, j] = pdf(mvnorm, [float(i), float(j)])
        end
    end
    mainbody_params = vcat(mainbody_loc, mainbody_var)
    return (mainbody_map, mainbody_params)
end

## Multiple Variable Node
@with_kw struct MultiVarNode <: MainbodyGen
    grid_dims::Tuple{Int64, Int64, Int64} = (50, 50, 1)
    mainbody_loc_bounds::Vector{Float64} = [15.0, 35.0]
    mainbody_var_min::Float64 = 20.0
    mainbody_var_max::Float64 = 40.0
    n_nodes::Int64 = 2
end

function Base.rand(rng::Random.AbstractRNG, mb::MultiVarNode)
    x_dim = mb.grid_dims[1]
    y_dim = mb.grid_dims[2]
    lode_map = zeros(Float64, x_dim, y_dim)
    mainbody_params = []
    mainbody_var = rand(rng)*(mb.mainbody_var_max - mb.mainbody_var_min) + mb.mainbody_var_min
    for _=1:mb.n_nodes
        d_lode_map = zeros(Float64, x_dim, y_dim)
        # mainbody_var = rand(rng)*(mb.mainbody_var_max - mb.mainbody_var_min) + mb.mainbody_var_min
        cov = Distributions.PDiagMat([mainbody_var, mainbody_var])
        mainbody_loc = rand(2).*(mb.mainbody_loc_bounds[2] - mb.mainbody_loc_bounds[1]) .+ mb.mainbody_loc_bounds[1]
        mvnorm = MvNormal(mainbody_loc, cov)
        for i = 1:x_dim
            for j = 1:y_dim
                d_lode_map[i, j] = pdf(mvnorm, [float(i), float(j)])
            end
        end
        d_mb_params = vcat(mainbody_loc, mainbody_var)
        lode_map += d_lode_map
        push!(mainbody_params, d_mb_params)
    end
    # lode_map ./= mb.n_nodes
    return (lode_map, mainbody_params)
end

Base.rand(mb::MultiVarNode) = rand(Random.GLOBAL_RNG, mb)

function perturb_sample(mb::MultiVarNode, mainbody_params::Vector, noise::Float64)
    x_dim = mb.grid_dims[1]
    y_dim = mb.grid_dims[2]
    lode_map = zeros(Float64, x_dim, y_dim)
    p_mainbody_params = []
    d_mb_var = 2.0*(rand() - 0.5)*noise
    for i=1:mb.n_nodes
        d_lode_map = zeros(Float64, x_dim, y_dim)
        mainbody_loc = mainbody_params[i][1:2]
        mainbody_var = mainbody_params[i][3]

        mainbody_loc += 2.0.*(rand(2) .- 0.5).*noise
        # mainbody_var += 20*(rand() - 0.5)*noise
        mainbody_var += d_mb_var
        mainbody_var = clamp(mainbody_var, 0.0, Inf)
        cov = Distributions.PDiagMat([mainbody_var, mainbody_var])
        mvnorm = MvNormal(mainbody_loc, cov)
        for i = 1:mb.grid_dims[1]
            for j = 1:mb.grid_dims[2]
                d_lode_map[i, j] = pdf(mvnorm, [float(i), float(j)])
            end
        end
        d_mb_params = vcat(mainbody_loc, mainbody_var)
        lode_map += d_lode_map
        push!(p_mainbody_params, d_mb_params)
    end
    # lode_map ./= mb.n_nodes
    return (lode_map, p_mainbody_params)
end
